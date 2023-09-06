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
#include "norm_matmul_op.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"

#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/matmul_operation.h"
#include "acltransformer/ops/cast_operation.h"


#include "kernels/funcs/format_utils.h"

namespace AclTransformer {
enum NormMatmulTensorId {
  IN_HIDDENSTATES = 0,
  IN_NORMWEIGHT,
  IN_NORMBIAS,
  IN_MATMULWEIGHT,

  OUT_NORMMATMULOUT,

  INTERMIDATE_CASTNORMWEIGHTOUT,
  INTERMIDATE_CASTNORMBIASOUT,
  INTERMIDATE_INPUTNORMOUT,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

NormMatmulOperation::NormMatmulOperation(const NormMatmulParam &param)
    : GraphOperation("NormMatmulOperation"), param_(param) {
  opGraph_.inTensorSize = IN_TENSOR_COUNT;
  opGraph_.outTensorSize = OUT_TENSOR_COUNT;
  opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
  opGraph_.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  GraphOperation::Node &astNormWeightNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &astNormBiasNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &matMulNode = opGraph_.nodes.at(nodeId++);

  astNormWeightNode.operation.reset(new AclTransformer::CastOperation());
  astNormWeightNode.inTensorIds = {IN_NORMWEIGHT};
  astNormWeightNode.outTensorIds = {INTERMIDATE_CASTNORMWEIGHTOUT};

  astNormBiasNode.operation.reset(new AclTransformer::CastOperation());
  astNormBiasNode.inTensorIds = {IN_NORMBIAS};
  astNormBiasNode.outTensorIds = {INTERMIDATE_CASTNORMBIASOUT};

  inputNormNode.operation.reset(
      new AclTransformer::NormOperation({param_.layerNormEps,
                                         param_.layerNormBeginNormAxis,
                                         param_.layerNormBeginNormAxis}));
  inputNormNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_CASTNORMWEIGHTOUT, INTERMIDATE_CASTNORMBIASOUT};
  inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

  matMulNode.operation.reset(
      new AclTransformer::MatmulOperation({param_.trans_x, param_.trans_y}));
  matMulNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_MATMULWEIGHT};
  matMulNode.outTensorIds = {OUT_NORMMATMULOUT};
}
NormMatmulOperation::~NormMatmulOperation() {}

uint64_t NormMatmulOperation::GetInTensorCount() const {
  return IN_TENSOR_COUNT;
}

uint64_t NormMatmulOperation::GetOutTensorCount() const {
  return OUT_TENSOR_COUNT;
}

AsdOps::Status NormMatmulOperation::InferShapeImpl(
    const AsdOps::SVector<AsdOps::Tensor> &inTensors,
    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const {
  // input * weight
  outTensorDescs.at(0).dtype = inTensors.at(IN_MATMULWEIGHT).desc.dtype;
  outTensorDescs.at(0).format = inTensors.at(IN_MATMULWEIGHT).desc.format;
  auto inTensorADims = inTensors.at(IN_HIDDENSTATES).desc.dims.size();
  auto inTensorBDims = inTensors.at(IN_MATMULWEIGHT).desc.dims.size();
  // 当前仅支持2维*2维，3维*3维，3维*2维
  if (inTensorADims == 3) {
    auto outTensorDim0 = inTensors.at(IN_HIDDENSTATES).desc.dims[0];
    auto outTensorDim1 =
        param_.trans_x
            ? inTensors.at(IN_HIDDENSTATES).desc.dims[inTensorADims - 1]
            : inTensors.at(IN_HIDDENSTATES).desc.dims[inTensorADims - 2];
    auto outTensorDim2 =
        param_.trans_y
            ? inTensors.at(IN_MATMULWEIGHT).desc.dims[inTensorBDims - 2]
            : inTensors.at(IN_MATMULWEIGHT).desc.dims[inTensorBDims - 1];
    outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
  } else {
    auto outTensorDim0 = param_.trans_x
                             ? inTensors.at(IN_HIDDENSTATES).desc.dims[1]
                             : inTensors.at(IN_HIDDENSTATES).desc.dims[0];
    auto outTensorDim1 = param_.trans_y
                             ? inTensors.at(IN_MATMULWEIGHT).desc.dims[0]
                             : inTensors.at(IN_MATMULWEIGHT).desc.dims[1];
    outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1};
  }

  return AsdOps::Status::OkStatus();
}

}  // namespace AclTransformer

NormMatmulWorkspace g_normMatmulWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::NormMatmulOperation> g_normMatmulOp;
std::unique_ptr<AclTransformer::Plan> g_normMatmulPlan;

std::vector<std::vector<int64_t>> NormMatmulOpInferShape(
    const std::vector<int64_t> &input_x_shape,
    const std::vector<int64_t> &weight_shape,
    const std::vector<int64_t> &bias_shape,
    const std::vector<int64_t> &input_y_shape,
    float epsilon,
    int begin_norm_axis,
    bool trans_x,
    bool trans_y) {
  std::vector<int64_t> x_shape = input_x_shape;
  std::vector<int64_t> y_shape = input_y_shape;

  auto ndims_x = x_shape.size();
  auto ndims_y = y_shape.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    x_shape.insert(x_shape.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    y_shape.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M, N;
  if (trans_x) {
    M = x_shape[ndims_x - 1];
  } else {
    M = x_shape[ndims_x - 2];
  }
  if (trans_y) {
    N = y_shape[ndims_y - 2];
  } else {
    N = y_shape[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(x_shape.begin(), x_shape.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(y_shape.begin(), y_shape.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(x_shape[i], y_shape[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }

  return {new_dims};
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
  if (workspaceSize <= g_normMatmulWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_normMatmulWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_normMatmulWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_normMatmulWorkSpace.workspace_);
    g_normMatmulWorkSpace.workspace_ = nullptr;
    g_normMatmulWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "NormMatmulOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_normMatmulWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("NormMatmulOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_normMatmulWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_normMatmulWorkSpace.workspace_; }

std::vector<paddle::Tensor> NormMatmulOp(const paddle::Tensor &input_x,
                                         const paddle::Tensor &weight,
                                         const paddle::Tensor &bias,
                                         const paddle::Tensor &input_y,
                                         float epsilon,
                                         int begin_norm_axis,
                                         bool trans_x,
                                         bool trans_y) {
  // std::cout << "run in NormMatmulOp" << std::endl;
  // std::cout << "trans x = " << trans_x << std::endl;
  // std::cout << "trans y = " << trans_y << std::endl;
  // std::cout << "begin norm axis = " << begin_norm_axis << std::endl;
  // std::cout << "epsilon  = " << epsilon << std::endl;

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_x_tensor =
      static_cast<const phi::DenseTensor *>(input_x.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor *>(weight.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor *>(bias.impl().get());
  auto input_y_tensor =
      static_cast<const phi::DenseTensor *>(input_y.impl().get());

  auto out_shape = NormMatmulOpInferShape(input_x.shape(),
                                          weight.shape(),
                                          bias.shape(),
                                          input_y.shape(),
                                          epsilon,
                                          begin_norm_axis,
                                          trans_x,
                                          trans_y)
                       .at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_x_tensor->dtype());
  if (!g_normMatmulOp) {
    AclTransformer::NormMatmulParam param = {
        epsilon, begin_norm_axis, trans_x, trans_y};
    g_normMatmulOp.reset(new AclTransformer::NormMatmulOperation(param));
    g_normMatmulPlan.reset(new AclTransformer::Plan);
    g_normMatmulOp->BuildPlan(g_normMatmulPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {
      input_x_tensor, weight_tensor, bias_tensor, input_y_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_normMatmulPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("normMatmulPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_normMatmulPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_normMatmulPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_normMatmulPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(norm_matmul_op)
    .Inputs({"X", "Scale", "Bias", "Y"})
    .Outputs({"Out"})
    .Attrs({
        "epsilon: float",
        "begin_norm_axis: int",
        "trans_x: bool",
        "trans_y: bool",
    })
    .SetKernelFn(PD_KERNEL(NormMatmulOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        NormMatmulOpInferShape));  // neccessary if the op has muti_inputs
#endif // PADDLE_WITH_ASCEND_TRANSFORMER_ACC