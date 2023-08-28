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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>

#include <iostream>
#include <vector>

#include "acltransformer/config.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/plan.h"
#include "gpt3_layer_op.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#include "self_attention/self_attention_without_kv_cache_gpt3_operation.h"

namespace AclTransformer {
enum GPT3LayerWithoutCacheDecoderParallelTensorId {
  IN_HIDDENSTATES_NOCACHE_PARALLEL = 0,
  IN_NORMWEIGHT_NOCACHE_PARALLEL,
  IN_NORMBIAS_NOCACHE_PARALLEL,
  IN_QKVMIXDWEIGHT_NOCACHE_PARALLEL,
  IN_QKVMIXDBIAS_NOCACHE_PARALLEL,
  IN_SELFOUTLINEARWEIGHT_NOCACHE_PARALLEL,
  IN_SELFOUTLINEARBIAS_NOCACHE_PARALLEL,
  IN_SELFOUTNORMWEIGHT_NOCACHE_PARALLEL,
  IN_SELFOUTNORMBIAS_NOCACHE_PARALLEL,
  IN_FFNLINEARWEIGHT_NOCACHE_PARALLEL,
  IN_FFNLINEARBIAS_NOCACHE_PARALLEL,
  IN_FFNOUTLINEARWEIGHT_NOCACHE_PARALLEL,
  IN_FFNOUTLINEARBIAS_NOCACHE_PARALLEL,
  IN_ATTENTIONMASK_NOCACHE_PARALLEL,

  OUT_GPT3LAYEROUT_NOCACHE_PARALLEL,
  OUT_PRESENTKEY_NOCACHE_PARALLEL,
  OUT_PRESENTVALUE_NOCACHE_PARALLEL,

  INTERMIDATE_INPUTNORMOUT_NOCACHE_PARALLEL,
  INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE_PARALLEL,
  INTERMIDATE_SELFOUT_NOCACHE_PARALLEL,
  INTERMIDATE_SELFLINEAROUT_NOCACHE_PARALLEL,
  INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE_PARALLEL,
  INTERMIDATE_SELFNORMOUT_NOCACHE_PARALLEL,
  INTERMIDATE_FFNOUT_NOCACHE_PARALLEL,
  INTERMIDATE_FFNLINEAROUT_NOCACHE_PARALLEL,
};

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

GPT3LayerWithoutCacheDecoderParallelOperation::
    GPT3LayerWithoutCacheDecoderParallelOperation(const GPT3LayerParam& param)
    : GraphOperation("GPT3LayerWithoutCacheDecoderParallelOperation"),
      param_(param) {
  opGraph_.inTensorSize = IN_TENSOR_COUNT;
  opGraph_.outTensorSize = OUT_TENSOR_COUNT;
  opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
  opGraph_.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  GraphOperation::Node& inputNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& selfOutLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& selfResidualAddNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& selfNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& ffnNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& ffnLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node& ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

  inputNormNode.operation.reset(
      new AclTransformer::NormOperation({param_.layerNormEps,
                                         param_.layerNormBeginNormAxis,
                                         param_.layerNormBeginNormAxis}));
  inputNormNode.inTensorIds = {IN_HIDDENSTATES_NOCACHE_PARALLEL,
                               IN_NORMWEIGHT_NOCACHE_PARALLEL,
                               IN_NORMBIAS_NOCACHE_PARALLEL};
  inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT_NOCACHE_PARALLEL};

  mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation(
      {false, false})); /* 加速库默认会将w进行转置 */
  mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT_NOCACHE_PARALLEL,
                                   IN_QKVMIXDWEIGHT_NOCACHE_PARALLEL,
                                   IN_QKVMIXDBIAS_NOCACHE_PARALLEL};
  mixdQkvLinearNode.outTensorIds = {
      INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE_PARALLEL};

  selfAttentionKvCacheNode.operation.reset(
      new AclTransformer::SelfAttentionWithoutKvCacheGPT3Operation(
          {false, param_.head_dim, param_.head_num, param_.layer_num}));
  selfAttentionKvCacheNode.inTensorIds = {
      INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE_PARALLEL,
      IN_ATTENTIONMASK_NOCACHE_PARALLEL};
  selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT_NOCACHE_PARALLEL,
                                           OUT_PRESENTKEY_NOCACHE_PARALLEL,
                                           OUT_PRESENTVALUE_NOCACHE_PARALLEL};

  selfOutLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
      {false, 0, 0, 0, "YES", "RowParallel", "hccl", true, param_.comm}));
  selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT_NOCACHE_PARALLEL,
                                   IN_SELFOUTLINEARWEIGHT_NOCACHE_PARALLEL,
                                   IN_SELFOUTLINEARBIAS_NOCACHE_PARALLEL};
  selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT_NOCACHE_PARALLEL};

  selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  selfResidualAddNode.inTensorIds = {
      IN_HIDDENSTATES_NOCACHE_PARALLEL,
      INTERMIDATE_SELFLINEAROUT_NOCACHE_PARALLEL};
  selfResidualAddNode.outTensorIds = {
      INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE_PARALLEL};

  selfNormNode.operation.reset(
      new AclTransformer::NormOperation({param_.layerNormEps}));
  selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE_PARALLEL,
                              IN_SELFOUTNORMWEIGHT_NOCACHE_PARALLEL,
                              IN_SELFOUTNORMBIAS_NOCACHE_PARALLEL};
  selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT_NOCACHE_PARALLEL};

  ffnNode.operation.reset(new AclTransformer::FfnOperation({false, false}));
  ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT_NOCACHE_PARALLEL,
                         IN_FFNLINEARWEIGHT_NOCACHE_PARALLEL,
                         IN_FFNLINEARBIAS_NOCACHE_PARALLEL};
  ffnNode.outTensorIds = {INTERMIDATE_FFNOUT_NOCACHE_PARALLEL};

  ffnLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
      {false, 0, 0, 0, "YES", "RowParallel", "hccl", true, param_.comm}));
  ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT_NOCACHE_PARALLEL,
                               IN_FFNOUTLINEARWEIGHT_NOCACHE_PARALLEL,
                               IN_FFNOUTLINEARBIAS_NOCACHE_PARALLEL};
  ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT_NOCACHE_PARALLEL};

  ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  ffnResidualAddNode.inTensorIds = {
      INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE_PARALLEL,
      INTERMIDATE_FFNLINEAROUT_NOCACHE_PARALLEL};
  ffnResidualAddNode.outTensorIds = {OUT_GPT3LAYEROUT_NOCACHE_PARALLEL};
}

GPT3LayerWithoutCacheDecoderParallelOperation::
    ~GPT3LayerWithoutCacheDecoderParallelOperation() {}

uint64_t GPT3LayerWithoutCacheDecoderParallelOperation::GetInTensorCount()
    const {
  return IN_TENSOR_COUNT;
}

uint64_t GPT3LayerWithoutCacheDecoderParallelOperation::GetOutTensorCount()
    const {
  return OUT_TENSOR_COUNT;
}

AsdOps::Status GPT3LayerWithoutCacheDecoderParallelOperation::InferShapeImpl(
    const AsdOps::SVector<AsdOps::Tensor>& inTensors,
    AsdOps::SVector<AsdOps::TensorDesc>& outTensorDescs) const {
  outTensorDescs.at(0) = inTensors.at(0).desc;

  outTensorDescs.at(1) = inTensors.at(0).desc;
  outTensorDescs.at(1).dims.at(2) = param_.head_num;
  outTensorDescs.at(1).dims.at(3) = param_.head_dim;

  outTensorDescs.at(2) = inTensors.at(0).desc;
  outTensorDescs.at(2).dims.at(2) = param_.head_num;
  outTensorDescs.at(2).dims.at(3) = param_.head_dim;
  return AsdOps::Status::OkStatus();
}

}  // namespace AclTransformer

GPT3LayerWorkspace g_gpt3WithoutCacheParallelWorkSpace = {nullptr, 0};
std::map<HcclComm, std::pair<std::shared_ptr<AclTransformer::GPT3LayerWithoutCacheDecoderParallelOperation>,
      std::shared_ptr<AclTransformer::Plan>>>
    g_gpt3WithoutDecoderParallelMap;

std::vector<std::vector<int64_t>> GPT3LayerWithoutCacheParallelOpInferShape(
    const std::vector<int64_t>& hidden_shape,
    const std::vector<int64_t>& norm_weight_shape,
    const std::vector<int64_t>& norm_bias_shape,
    const std::vector<int64_t>& mix_linear_weight_shape,
    const std::vector<int64_t>& mix_linear_bias_shape,
    const std::vector<int64_t>& self_out_linear_weight_shape,
    const std::vector<int64_t>& self_out_linear_bias_shape,
    const std::vector<int64_t>& self_out_norm_weight_shape,
    const std::vector<int64_t>& self_out_norm_bias_shape,
    const std::vector<int64_t>& ffn_linear_weight_shape,
    const std::vector<int64_t>& ffn_linear_bias_shape,
    const std::vector<int64_t>& ffn_out_linear_weight_shape,
    const std::vector<int64_t>& ffn_out_linear_bias_shape,
    const std::vector<int64_t>& attention_mask_shape,
    int begin_norm_axis,
    float epsilon,
    std::vector<int32_t> shape,
    float scale) {
  int32_t head_dim = shape[3] / 3;
  int32_t head_num = self_out_linear_weight_shape[0] / head_dim;
  
  std::vector<int64_t> presentkey_shape;   /* [bs, seq_len, hidden_size] */
  std::vector<int64_t> presentvalue_shape; /* [bs, seq_len, hidden_size] */
  presentkey_shape.push_back(hidden_shape.at(0));
  presentkey_shape.push_back(hidden_shape.at(1));
  presentkey_shape.push_back(head_num); /* TODO:当前写死了6.7B模型  */
  presentkey_shape.push_back(head_dim);

  presentvalue_shape.push_back(hidden_shape.at(0));
  presentvalue_shape.push_back(hidden_shape.at(1));
  presentvalue_shape.push_back(head_num);
  presentvalue_shape.push_back(head_dim);

  return {hidden_shape, presentkey_shape, presentvalue_shape};
}

static void BuildVariantPack(std::vector<const phi::DenseTensor*>& inTensors,
                             std::vector<const phi::DenseTensor*>& outTensors,
                             AclTransformer::VariantPack& variantPack) {
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
  if (workspaceSize <= g_gpt3WithoutCacheParallelWorkSpace.workspaceSize_) {
    return;
  }
  if (g_gpt3WithoutCacheParallelWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_gpt3WithoutCacheParallelWorkSpace.workspace_);
    g_gpt3WithoutCacheParallelWorkSpace.workspace_ = nullptr;
    g_gpt3WithoutCacheParallelWorkSpace.workspaceSize_ = 0;
  }

  int st =
      AsdRtMemMallocDevice((void**)&(g_gpt3WithoutCacheParallelWorkSpace.workspace_),
                           workspaceSize,
                           ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External(
          "GPT3LayerWithoutCacheParallelOp SetWorkspace AsdRtMemMallocDevice,"
          "fail, ret: %d .",
          st));

  g_gpt3WithoutCacheParallelWorkSpace.workspaceSize_ = workspaceSize;
}

static void* GetWorkspace() { return g_gpt3WithoutCacheParallelWorkSpace.workspace_; }

void GPT3LayerWithoutCacheParallelGetTensorInputs(
    const paddle::Tensor& hidden,
    const paddle::Tensor& norm_weight,
    const paddle::Tensor& norm_bias,
    const paddle::Tensor& mix_linear_weight,
    const paddle::Tensor& mix_linear_bias,
    const paddle::Tensor& self_out_linear_weight,
    const paddle::Tensor& self_out_linear_bias,
    const paddle::Tensor& self_out_norm_weight,
    const paddle::Tensor& self_out_norm_bias,
    const paddle::Tensor& ffn_linear_weight,
    const paddle::Tensor& ffn_linear_bias,
    const paddle::Tensor& ffn_out_linear_weight,
    const paddle::Tensor& ffn_out_linear_bias,
    const paddle::Tensor& attention_mask,
    std::vector<const phi::DenseTensor*>& inputs) {
  auto hidden_tensor = static_cast<const phi::DenseTensor*>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor*>(norm_weight.impl().get());
  auto norm_bias_tensor = static_cast<const phi::DenseTensor*>(norm_bias.impl().get());
  auto mix_linear_weight_tensor = static_cast<phi::DenseTensor*>(mix_linear_weight.impl().get());
  auto mix_linear_bias_tensor = static_cast<const phi::DenseTensor*>(mix_linear_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<phi::DenseTensor*>(self_out_linear_weight.impl().get());
  auto self_out_linear_bias_tensor = static_cast<const phi::DenseTensor*>(self_out_linear_bias.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor*>(self_out_norm_weight.impl().get());
  auto self_out_norm_bias_tensor = static_cast<const phi::DenseTensor*>(self_out_norm_bias.impl().get());
  auto ffn_linear_weight_tensor = static_cast<phi::DenseTensor*>(ffn_linear_weight.impl().get());
  auto ffn_linear_bias_tensor = static_cast<const phi::DenseTensor*>(ffn_linear_bias.impl().get());
  auto ffn_out_linear_weight_tensor = static_cast<phi::DenseTensor*>(ffn_out_linear_weight.impl().get());
  auto ffn_out_linear_bias_tensor = static_cast<const phi::DenseTensor*>(ffn_out_linear_bias.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor*>(attention_mask.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_bias_tensor);
  phi::DDim dims1 = mix_linear_weight_tensor->dims();
  inputs.push_back(&mix_linear_weight_tensor->Resize(phi::make_ddim({dims1[1], dims1[0]})));
  inputs.push_back(mix_linear_bias_tensor);
  phi::DDim dims2 = self_out_linear_weight_tensor->dims();
  inputs.push_back(&self_out_linear_weight_tensor->Resize(phi::make_ddim({dims2[1], dims2[0]})));
  inputs.push_back(self_out_linear_bias_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_bias_tensor);
  phi::DDim dims3 = ffn_linear_weight_tensor->dims();
  inputs.push_back(&ffn_linear_weight_tensor->Resize(phi::make_ddim({dims3[1], dims3[0]})));
  inputs.push_back(ffn_linear_bias_tensor);
  phi::DDim dims4 = ffn_out_linear_weight_tensor->dims();
  inputs.push_back(&ffn_out_linear_weight_tensor->Resize(phi::make_ddim({dims4[1], dims4[0]})));
  inputs.push_back(ffn_out_linear_bias_tensor);
  inputs.push_back(attention_mask_tensor);
}

std::vector<paddle::Tensor> GPT3LayerWithoutCacheParallelOp(
    const paddle::Tensor& hidden,
    const paddle::Tensor& norm_weight,
    const paddle::Tensor& norm_bias,
    const paddle::Tensor& mix_linear_weight,
    const paddle::Tensor& mix_linear_bias,
    const paddle::Tensor& self_out_linear_weight,
    const paddle::Tensor& self_out_linear_bias,
    const paddle::Tensor& self_out_norm_weight,
    const paddle::Tensor& self_out_norm_bias,
    const paddle::Tensor& ffn_linear_weight,
    const paddle::Tensor& ffn_linear_bias,
    const paddle::Tensor& ffn_out_linear_weight,
    const paddle::Tensor& ffn_out_linear_bias,
    const paddle::Tensor& attention_mask,
    int begin_norm_axis,
    float epsilon,
    std::vector<int32_t> shape,
    float scale) {
  int32_t layer_num = (int32_t)scale;
  int32_t head_dim = shape[3] / 3;
  int32_t head_num = self_out_linear_weight.shape()[0] / head_dim;

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};
  HcclComm comm = reinterpret_cast<HcclComm>(dev_ctx->xccl_comm());

  std::vector<const phi::DenseTensor*> inputs;
  GPT3LayerWithoutCacheParallelGetTensorInputs(hidden,
                                               norm_weight,
                                               norm_bias,
                                               mix_linear_weight,
                                               mix_linear_bias,
                                               self_out_linear_weight,
                                               self_out_linear_bias,
                                               self_out_norm_weight,
                                               self_out_norm_bias,
                                               ffn_linear_weight,
                                               ffn_linear_bias,
                                               ffn_out_linear_weight,
                                               ffn_out_linear_bias,
                                               attention_mask,
                                               inputs);

  std::vector<int64_t> hidden_shape = hidden.shape();
  std::vector<int64_t> presentkey_shape;
  std::vector<int64_t> presentvalue_shape;

  presentkey_shape.push_back(hidden_shape.at(0));
  presentkey_shape.push_back(hidden_shape.at(1));
  presentkey_shape.push_back(head_num);
  presentkey_shape.push_back(head_dim);

  presentvalue_shape.push_back(hidden_shape.at(0));
  presentvalue_shape.push_back(hidden_shape.at(1));
  presentvalue_shape.push_back(head_num);
  presentvalue_shape.push_back(head_dim);

  std::shared_ptr<phi::DenseTensor> gpt3layerout_tensor =
      std::make_shared<phi::DenseTensor>();
  gpt3layerout_tensor->Resize(phi::make_ddim(hidden_shape));
  dev_ctx->Alloc(gpt3layerout_tensor.get(), inputs.at(0)->dtype());

  std::shared_ptr<phi::DenseTensor> presentkey_tensor =
      std::make_shared<phi::DenseTensor>();
  presentkey_tensor->Resize(phi::make_ddim(presentkey_shape));
  dev_ctx->Alloc(presentkey_tensor.get(), inputs.at(0)->dtype());

  std::shared_ptr<phi::DenseTensor> presentvalue_tensor =
      std::make_shared<phi::DenseTensor>();
  presentvalue_tensor->Resize(phi::make_ddim(presentvalue_shape));
  dev_ctx->Alloc(presentvalue_tensor.get(), inputs.at(0)->dtype());

  std::vector<const phi::DenseTensor*> outputs;
  outputs.push_back(gpt3layerout_tensor.get());
  outputs.push_back(presentkey_tensor.get());
  outputs.push_back(presentvalue_tensor.get());

  std::shared_ptr<AclTransformer::GPT3LayerWithoutCacheDecoderParallelOperation>
      gpt3WithoutDecoderParallelOp;
  std::shared_ptr<AclTransformer::Plan> gpt3WithoutCacheParallelPlan;

  auto it = g_gpt3WithoutDecoderParallelMap.find(comm);
  if (it == g_gpt3WithoutDecoderParallelMap.end()) {
    std::cout << "GPT3LayerWithoutCacheParallelOp comm: " << comm << std::endl;
    std::shared_ptr<AclTransformer::GPT3LayerWithoutCacheDecoderParallelOperation>
        decoderOp;
    std::shared_ptr<AclTransformer::Plan> plan;

    AclTransformer::GPT3LayerParam param = {
        epsilon, begin_norm_axis, head_dim, head_num, layer_num, {}, {}, comm};
    decoderOp.reset(new AclTransformer::GPT3LayerWithoutCacheDecoderParallelOperation(param));
    plan.reset(new AclTransformer::Plan);
    decoderOp->BuildPlan(plan.get());
    g_gpt3WithoutDecoderParallelMap[comm] = std::make_pair(decoderOp, plan);
    gpt3WithoutDecoderParallelOp = decoderOp;
    gpt3WithoutCacheParallelPlan = plan;
  } else {
    gpt3WithoutDecoderParallelOp = it->second.first;
    gpt3WithoutCacheParallelPlan = it->second.second;
  }
  AclTransformer::VariantPack variantPack;
  BuildVariantPack(inputs, outputs, variantPack);

  /* Set up */
  AsdOps::Status st =
      gpt3WithoutCacheParallelPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("GPT3LayerWithoutCacheParallelOp Setup plan failed,"
                            "ret message: %s .",
                            st.Message()));

  variantPack.workspaceSize =
      gpt3WithoutCacheParallelPlan->GetWorkspaceSize();

  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  /* Execute */
  st = gpt3WithoutCacheParallelPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External(
                        "GPT3LayerWithoutCacheParallelOp Execute plan failed,"
                        "ret message: %s .",
                        st.Message()));
  static uint64_t executeCount_ = 0;
  executeCount_++;
  if ((executeCount_) % layer_num == 0) {  // 1.....32,第32次同步
    int ret = aclrtSynchronizeStream(stream);
  }
  return {paddle::Tensor(gpt3layerout_tensor),
          paddle::Tensor(presentkey_tensor),
          paddle::Tensor(presentvalue_tensor)};
}

PD_BUILD_OP(gpt3_layer_without_kvcache_parallel)
    .Inputs({"Hidden",
             "NormWeight",
             "NormBias",
             "MixLinearWeight",
             "MixLinearBias",
             "SelfOutLinearWeight",
             "SelfOutLinearBias",
             "SelfOutNormWeight",
             "SelfOutNormBias",
             "FfnLinearWeight",
             "FfnLinearBias",
             "FfnOutLinearWeight",
             "FfnOutLinearBias",
             "AttentionMask"})
    .Outputs({"Out", "PresentKey", "PresentValue"})
    .Attrs({"begin_norm_axis: int",
            "epsilon: float",
            "shape: std::vector<int>",
            "scale: float"})
    .SetKernelFn(PD_KERNEL(GPT3LayerWithoutCacheParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(GPT3LayerWithoutCacheParallelOpInferShape));

#endif
