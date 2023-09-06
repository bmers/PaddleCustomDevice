// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

#include "kernels/funcs/format_utils.h"
#include "acltransformer/params/norm.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"
#include "acltransformer/ops/embedding_operation.h"
#include "kernels/funcs/format_utils.h"


struct EmbeddingWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

EmbeddingWorkspace g_embeddingWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::EmbeddingOperation> g_embeddingOp;
std::unique_ptr<AclTransformer::Plan> g_embeddingPlan;

std::vector<std::vector<int64_t>> EmbeddingOpInferShape(
    const std::vector<int64_t> &input_w_shape,
    const std::vector<int64_t> &input_ids_shape,
    int64_t padding_idx) {
  std::vector<int64_t> w_shape = input_w_shape;
  std::vector<int64_t> ids_shape = input_ids_shape;

  std::vector<int64_t> out_dims;
  out_dims.assign(ids_shape.begin(), ids_shape.end());
  out_dims.push_back(w_shape[1]);

  return {out_dims};
}

static void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                             std::vector<const phi::DenseTensor *> &outTensors,
                             AclTransformer::VariantPack &variantPack) {
  variantPack.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); ++i) {
    variantPack.inTensors.at(i) =
        ConvertDenseTensorToAsdTensor(*(inTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }

  variantPack.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); ++i) {
    variantPack.outTensors.at(i) =
        ConvertDenseTensorToAsdTensor(*(outTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.outTensors.at(i).desc.format ==
            AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.outTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }
}

static void SetWorkspace(uint64_t workspaceSize) {
  if (workspaceSize <= g_embeddingWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_embeddingWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_embeddingWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_embeddingWorkSpace.workspace_);
    g_embeddingWorkSpace.workspace_ = nullptr;
    g_embeddingWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "embeddingOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_embeddingWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("EmbeddingOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_embeddingWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_embeddingWorkSpace.workspace_; }
constexpr int64_t kNoPadding = -1;

std::vector<paddle::Tensor> EmbeddingOp(const paddle::Tensor &input_w,
                                         const paddle::Tensor &input_ids,
                                         int64_t padding_idx) {
  // std::cout << "run in EnbeddingOp" << std::endl;
  // std::cout << "padding_idx = " << padding_idx << std::endl;

  if (padding_idx == kNoPadding) {
      padding_idx = 0;
  } else {
      padding_idx =
          padding_idx < 0 ? padding_idx + input_w.dims()[0] : padding_idx;
  }

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_w_tensor =
      static_cast<const phi::DenseTensor *>(input_w.impl().get());
  auto input_ids_tensor =
      static_cast<const phi::DenseTensor *>(input_ids.impl().get());

  auto out_shape = EmbeddingOpInferShape(input_w.shape(),
                                          input_ids.shape(),
                                          padding_idx).at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_w_tensor->dtype());
  if (!g_embeddingOp) {
    AclTransformer::EmbeddingParam param = {padding_idx};
    g_embeddingOp.reset(new AclTransformer::EmbeddingOperation(param));
    g_embeddingPlan.reset(new AclTransformer::Plan);
    g_embeddingOp->BuildPlan(g_embeddingPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {
      input_w_tensor, input_ids_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_embeddingPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("embeddingPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_embeddingPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_embeddingPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_embeddingPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(embedding_op)
    .Inputs({"W", "Ids"})
    .Outputs({"Out"})
    .Attrs({"padding_idx:int64_t"})
    .SetKernelFn(PD_KERNEL(EmbeddingOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        EmbeddingOpInferShape));  // neccessary if the op has muti_inputs
#endif