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
#include "embedding_add_op.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"

#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "kernels/funcs/format_utils.h"

namespace AclTransformer {
enum PretreatmentTensorId {
  IN_EMBEDDINGW0 = 0,
  IN_EMBEDDINGIDS0,
  IN_EMBEDDINGW1,
  IN_EMBEDDINGIDS1,

  OUT_ADDOUT,

  INTERMIDATE_EMBEDDING0,
  INTERMIDATE_EMBEDDING1,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;

PretreatmentOperation::PretreatmentOperation(const PretreatmentParam &param)
    : GraphOperation("PretreatmentOperation"), param_(param) {
  opGraph_.inTensorSize = IN_TENSOR_COUNT;
  opGraph_.outTensorSize = OUT_TENSOR_COUNT;
  opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
  opGraph_.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  GraphOperation::Node &inputEmbedding0Node = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &inputEmbedding1Node = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &addNode = opGraph_.nodes.at(nodeId++);

  inputEmbedding0Node.operation.reset(new AclTransformer::EmbeddingOperation({param_.axis0}));
  inputEmbedding0Node.inTensorIds = {IN_EMBEDDINGW0, IN_EMBEDDINGIDS0};
  inputEmbedding0Node.outTensorIds = {INTERMIDATE_EMBEDDING0};

  inputEmbedding1Node.operation.reset(new AclTransformer::EmbeddingOperation({param_.axis1}));
  inputEmbedding1Node.inTensorIds = {IN_EMBEDDINGW1, IN_EMBEDDINGIDS1};
  inputEmbedding1Node.outTensorIds = {INTERMIDATE_EMBEDDING1};

  addNode.operation.reset(new AclTransformer::AddOperation({1}));
  addNode.inTensorIds = {INTERMIDATE_EMBEDDING0, INTERMIDATE_EMBEDDING1};
  addNode.outTensorIds = {OUT_ADDOUT};
}
PretreatmentOperation::~PretreatmentOperation() {}

uint64_t PretreatmentOperation::GetInTensorCount() const {
  return IN_TENSOR_COUNT;
}

uint64_t PretreatmentOperation::GetOutTensorCount() const {
  return OUT_TENSOR_COUNT;
}

AsdOps::Status PretreatmentOperation::InferShapeImpl(
    const AsdOps::SVector<AsdOps::Tensor> &inTensors,
    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const {
  outTensorDescs.at(0).dtype = inTensors.at(IN_EMBEDDINGW0).desc.dtype;
  outTensorDescs.at(0).format = inTensors.at(IN_EMBEDDINGW0).desc.format;
  outTensorDescs.at(0).dims = {inTensors.at(IN_EMBEDDINGIDS0).desc.dims[0],
                               inTensors.at(IN_EMBEDDINGIDS0).desc.dims[1],
                               inTensors.at(IN_EMBEDDINGW0).desc.dims[1]};

  return AsdOps::Status::OkStatus();
}
}  // namespace AclTransformer

PretreatmentWorkspace g_pretreatmentWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::PretreatmentOperation> g_pretreatmentOp;
std::unique_ptr<AclTransformer::Plan> g_pretreatmentPlan;

std::vector<std::vector<int64_t>> PretreatmentOpInferShape(
    const std::vector<int64_t> &input_w0_shape,
    const std::vector<int64_t> &input_ids0_shape,
    const std::vector<int64_t> &input_w1_shape,
    const std::vector<int64_t> &input_ids1_shape,
    int64_t padding_idx0,
    int64_t padding_idx1) {
  std::vector<int64_t> w_shape = input_w0_shape;
  std::vector<int64_t> ids_shape = input_ids0_shape;

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
  if (workspaceSize <= g_pretreatmentWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_pretreatmentWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_pretreatmentWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_pretreatmentWorkSpace.workspace_);
    g_pretreatmentWorkSpace.workspace_ = nullptr;
    g_pretreatmentWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "PretreatmentOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_pretreatmentWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("PretreatmentOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_pretreatmentWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_pretreatmentWorkSpace.workspace_; }
constexpr int64_t kNoPadding = -1;

std::vector<paddle::Tensor> PretreatmentOp(const paddle::Tensor &input_w0,
                                         const paddle::Tensor &input_ids0,
                                         const paddle::Tensor &input_w1,
                                         const paddle::Tensor &input_ids1,
                                         int64_t padding_idx0,
                                         int64_t padding_idx1) {
  // std::cout << "run in PretreatmentOp" << std::endl;
  // std::cout << "padding_idx0 = " << padding_idx0 << "padding_idx1 = " << padding_idx1 << std::endl;

  if (padding_idx0 == kNoPadding) {
      padding_idx0 = 0;
  } else {
      padding_idx0 =
          padding_idx0 < 0 ? padding_idx0 + input_w0.dims()[0] : padding_idx0;
  }
  if (padding_idx1 == kNoPadding) {
      padding_idx1 = 0;
  } else {
      padding_idx1 =
          padding_idx1 < 0 ? padding_idx1 + input_w1.dims()[0] : padding_idx1;
  }

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_ids0.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_w0_tensor =
      static_cast<const phi::DenseTensor *>(input_w0.impl().get());
  auto input_ids0_tensor =
      static_cast<const phi::DenseTensor *>(input_ids0.impl().get());
  auto input_w1_tensor =
      static_cast<const phi::DenseTensor *>(input_w1.impl().get());
  auto input_ids1_tensor =
      static_cast<const phi::DenseTensor *>(input_ids1.impl().get());

  auto out_shape = PretreatmentOpInferShape(input_w0.shape(),
                                          input_ids0.shape(),
                                          input_w1.shape(),
                                          input_ids1.shape(),
                                          padding_idx0,
                                          padding_idx1).at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_w0_tensor->dtype());
  if (!g_pretreatmentOp) {
    AclTransformer::PretreatmentParam param = {padding_idx0, padding_idx1};
    g_pretreatmentOp.reset(new AclTransformer::PretreatmentOperation(param));
    g_pretreatmentPlan.reset(new AclTransformer::Plan);
    g_pretreatmentOp->BuildPlan(g_pretreatmentPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {
      input_w0_tensor, input_ids0_tensor, input_w1_tensor, input_ids1_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_pretreatmentPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("pretreatmentPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_pretreatmentPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_pretreatmentPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_pretreatmentPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(pretreatment_op)
    .Inputs({"W0", "Ids0", "W1", "Ids1"})
    .Outputs({"Out"})
    .Attrs({
        "padding_idx0: int64_t",
        "padding_idx1: int64_t",
    })
    .SetKernelFn(PD_KERNEL(PretreatmentOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        PretreatmentOpInferShape));  // neccessary if the op has muti_inputs
#endif // PADDLE_WITH_ASCEND_TRANSFORMER_ACC