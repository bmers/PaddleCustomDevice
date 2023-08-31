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
#include <iostream>
#include <vector>
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

#include "gpt3_layer_op.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/cast_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "self_attention/self_attention_without_kv_cache_gpt3_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "kernels/funcs/format_utils.h"

namespace AclTransformer {
enum GPT3LayerWithoutCacheDecoderTensorId {
  IN_HIDDENSTATES_NOCACHE = 0,
  IN_NORMWEIGHT_NOCACHE,
  IN_NORMBIAS_NOCACHE,
  IN_QKVMIXDWEIGHT_NOCACHE,
  IN_QKVMIXDBIAS_NOCACHE,
  IN_SELFOUTLINEARWEIGHT_NOCACHE,
  IN_SELFOUTLINEARBIAS_NOCACHE,
  IN_SELFOUTNORMWEIGHT_NOCACHE,
  IN_SELFOUTNORMBIAS_NOCACHE,
  IN_FFNLINEARWEIGHT_NOCACHE,
  IN_FFNLINEARBIAS_NOCACHE,
  IN_FFNOUTLINEARWEIGHT_NOCACHE,
  IN_FFNOUTLINEARBIAS_NOCACHE,
  IN_ATTENTIONMASK_NOCACHE,

  OUT_GPT3LAYEROUT_NOCACHE,
  OUT_PRESENTKEY_NOCACHE,
  OUT_PRESENTVALUE_NOCACHE,

  INTERMIDATE_INPUTNORMOUT_NOCACHE,
  INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE,
  INTERMIDATE_SELFOUT_NOCACHE,
  INTERMIDATE_SELFLINEAROUT_NOCACHE,
  INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE,

  INTERMIDATE_SELFNORMOUT_NOCACHE,
  INTERMIDATE_FFNOUT_NOCACHE,
  INTERMIDATE_FFNLINEAROUT_NOCACHE,
};

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

GPT3LayerWithoutCacheDecoderOperation::GPT3LayerWithoutCacheDecoderOperation(const GPT3LayerParam &param)
    : GraphOperation("GPT3LayerWithoutCacheDecoderOperation"), param_(param)
{
  opGraph_.inTensorSize = IN_TENSOR_COUNT;
  opGraph_.outTensorSize = OUT_TENSOR_COUNT;
  opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
  opGraph_.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

  inputNormNode.operation.reset(new AclTransformer::NormOperation(
                      {param_.layerNormEps, param_.layerNormBeginNormAxis, param_.layerNormBeginNormAxis}));
  inputNormNode.inTensorIds = {IN_HIDDENSTATES_NOCACHE, IN_NORMWEIGHT_NOCACHE, IN_NORMBIAS_NOCACHE};
  inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT_NOCACHE};

  mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true})); /* 加速库默认会将w进行转置 */
  mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT_NOCACHE, IN_QKVMIXDWEIGHT_NOCACHE, IN_QKVMIXDBIAS_NOCACHE};
  mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE};
    
  selfAttentionKvCacheNode.operation.reset(new AclTransformer::SelfAttentionWithoutKvCacheGPT3Operation(
          {false, param_.head_dim, param_.head_num, param_.layer_num}));
              selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV_NOCACHE,
                IN_ATTENTIONMASK_NOCACHE};
  selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT_NOCACHE, OUT_PRESENTKEY_NOCACHE, OUT_PRESENTVALUE_NOCACHE};
                        
  selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true}));
  selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT_NOCACHE, IN_SELFOUTLINEARWEIGHT_NOCACHE, IN_SELFOUTLINEARBIAS_NOCACHE};
  selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT_NOCACHE};
                                    
  selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES_NOCACHE, INTERMIDATE_SELFLINEAROUT_NOCACHE};
  selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE};

  selfNormNode.operation.reset(new AclTransformer::NormOperation(
      {param_.layerNormEps, param_.layerNormBeginNormAxis, param_.layerNormBeginNormAxis}));
  selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE, IN_SELFOUTNORMWEIGHT_NOCACHE, IN_SELFOUTNORMBIAS_NOCACHE};
  selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT_NOCACHE};
          
  ffnNode.operation.reset(new AclTransformer::FfnOperation({false, true}));
  ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT_NOCACHE, IN_FFNLINEARWEIGHT_NOCACHE, IN_FFNLINEARBIAS_NOCACHE};
  ffnNode.outTensorIds = {INTERMIDATE_FFNOUT_NOCACHE};
  
  ffnLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true}));
  ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT_NOCACHE, IN_FFNOUTLINEARWEIGHT_NOCACHE, IN_FFNOUTLINEARBIAS_NOCACHE};
  ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT_NOCACHE};
          
  ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_NOCACHE, INTERMIDATE_FFNLINEAROUT_NOCACHE};
  ffnResidualAddNode.outTensorIds = {OUT_GPT3LAYEROUT_NOCACHE};
}
                                
GPT3LayerWithoutCacheDecoderOperation::~GPT3LayerWithoutCacheDecoderOperation() {}

uint64_t GPT3LayerWithoutCacheDecoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GPT3LayerWithoutCacheDecoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GPT3LayerWithoutCacheDecoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
  outTensorDescs.at(0) = inTensors.at(0).desc;

  outTensorDescs.at(1) = inTensors.at(0).desc;
  outTensorDescs.at(1).dims.at(2) = param_.head_num;
  outTensorDescs.at(1).dims.at(3) = param_.head_dim;

  outTensorDescs.at(2) = inTensors.at(0).desc;
  outTensorDescs.at(2).dims.at(2) = param_.head_num;
  outTensorDescs.at(2).dims.at(3) = param_.head_dim;
  return AsdOps::Status::OkStatus();
}

} // namespace AclTransformer

GPT3LayerWorkspace g_gpt3WithoutCacheWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::GPT3LayerWithoutCacheDecoderOperation> g_gpt3WithoutDecoderOp;
std::unique_ptr<AclTransformer::Plan> g_gpt3WithoutCachePlan;

std::vector<std::vector<int64_t>> GPT3LayerWithoutCacheOpInferShape(
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
  int32_t head_num = hidden_shape[2] / head_dim;
  
  std::vector<int64_t> presentkey_shape; /* [bs, seq_len, hidden_size] */
  std::vector<int64_t> presentvalue_shape; /* [bs, seq_len, hidden_size] */
  presentkey_shape.push_back(hidden_shape.at(0));
  presentkey_shape.push_back(hidden_shape.at(1));
  presentkey_shape.push_back(head_num);  /* TODO:当前写死了6.7B模型  */
  presentkey_shape.push_back(head_dim);
            
  presentvalue_shape.push_back(hidden_shape.at(0));
  presentvalue_shape.push_back(hidden_shape.at(1));
  presentvalue_shape.push_back(head_num);
  presentvalue_shape.push_back(head_dim);
                    
  return {hidden_shape, presentkey_shape, presentvalue_shape};
}

static void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                             std::vector<const phi::DenseTensor *> &outTensors,
                             AclTransformer::VariantPack &variantPack)
{
  variantPack.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); ++i) {
    variantPack.inTensors.at(i) = ConvertDenseTensorToAsdTensor(*(inTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
        variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }

  variantPack.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); ++i) {
      variantPack.outTensors.at(i) = ConvertDenseTensorToAsdTensor(*(outTensors.at(i)));
      if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.outTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
          variantPack.outTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
      }
  }
}

static void SetWorkspace(uint64_t workspaceSize)
{
  if (workspaceSize <= g_gpt3WithoutCacheWorkSpace.workspaceSize_) {
          return;
  }
  if(g_gpt3WithoutCacheWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_gpt3WithoutCacheWorkSpace.workspace_);
    g_gpt3WithoutCacheWorkSpace.workspace_ = nullptr;
    g_gpt3WithoutCacheWorkSpace.workspaceSize_ = 0;
  }

  int st = AsdRtMemMallocDevice((void **)&(g_gpt3WithoutCacheWorkSpace.workspace_), workspaceSize, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(st,
                    ASDRT_SUCCESS,
                    phi::errors::External(
                    "GPT3LayerWithoutCacheOp SetWorkspace AsdRtMemMallocDevice,"
                    "fail, ret: %d .", st));

  g_gpt3WithoutCacheWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_gpt3WithoutCacheWorkSpace.workspace_;}
                   
void GPT3LayerWithoutCacheGetTensorInputs(const paddle::Tensor& hidden,
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
  auto mix_linear_weight_tensor = static_cast<const phi::DenseTensor*>(mix_linear_weight.impl().get());
  auto mix_linear_bias_tensor = static_cast<const phi::DenseTensor*>(mix_linear_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<const phi::DenseTensor*>(self_out_linear_weight.impl().get());
  auto self_out_linear_bias_tensor = static_cast<const phi::DenseTensor*>(self_out_linear_bias.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor*>(self_out_norm_weight.impl().get());
  auto self_out_norm_bias_tensor = static_cast<const phi::DenseTensor*>(self_out_norm_bias.impl().get());
  auto ffn_linear_weight_tensor = static_cast<const phi::DenseTensor*>(ffn_linear_weight.impl().get());
  auto ffn_linear_bias_tensor = static_cast<const phi::DenseTensor*>(ffn_linear_bias.impl().get());
  auto ffn_out_linear_weight_tensor = static_cast<const phi::DenseTensor*>(ffn_out_linear_weight.impl().get());
  auto ffn_out_linear_bias_tensor = static_cast<const phi::DenseTensor*>(ffn_out_linear_bias.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor*>(attention_mask.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_bias_tensor);
  inputs.push_back(mix_linear_weight_tensor);
  inputs.push_back(mix_linear_bias_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_bias_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_bias_tensor);
  inputs.push_back(ffn_linear_weight_tensor);
  inputs.push_back(ffn_linear_bias_tensor);
  inputs.push_back(ffn_out_linear_weight_tensor);
  inputs.push_back(ffn_out_linear_bias_tensor);
  inputs.push_back(attention_mask_tensor);
}

std::vector<paddle::Tensor> GPT3LayerWithoutCacheOp(const paddle::Tensor& hidden,
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
  int32_t head_num = hidden.shape()[2] / head_dim;

  auto dev_ctx = static_cast<const phi::CustomContext*>(
  paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  std::vector<const phi::DenseTensor*> inputs;
  GPT3LayerWithoutCacheGetTensorInputs(hidden,
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

  // not use this.
  auto out_shape = GPT3LayerWithoutCacheOpInferShape(hidden.shape(),
                                                     norm_weight.shape(),
                                                     norm_bias.shape(),
                                                     mix_linear_weight.shape(),
                                                     mix_linear_bias.shape(),
                                                     self_out_linear_weight.shape(),
                                                     self_out_linear_bias.shape(),
                                                     self_out_norm_weight.shape(),
                                                     self_out_norm_bias.shape(),
                                                     ffn_linear_weight.shape(), 
                                                     ffn_linear_bias.shape(),
                                                     ffn_out_linear_weight.shape(),
                                                     ffn_out_linear_bias.shape(),
                                                     attention_mask.shape(),
													                           begin_norm_axis,
                                                     epsilon,
                                                     shape,
                                                     scale);

  std::shared_ptr<phi::DenseTensor> gpt3layerout_tensor =
      std::make_shared<phi::DenseTensor>();
  gpt3layerout_tensor->Resize(phi::make_ddim(out_shape.at(0)));
  dev_ctx->Alloc(gpt3layerout_tensor.get(), inputs.at(0)->dtype());

  std::shared_ptr<phi::DenseTensor> presentkey_tensor =
      std::make_shared<phi::DenseTensor>();
  presentkey_tensor->Resize(phi::make_ddim(out_shape.at(1)));
  dev_ctx->Alloc(presentkey_tensor.get(), inputs.at(0)->dtype());

  std::shared_ptr<phi::DenseTensor> presentvalue_tensor =
  std::make_shared<phi::DenseTensor>();
  presentvalue_tensor->Resize(phi::make_ddim(out_shape.at(2)));
  dev_ctx->Alloc(presentvalue_tensor.get(), inputs.at(0)->dtype());
                     
  std::vector<const phi::DenseTensor*> outputs;
  outputs.push_back(gpt3layerout_tensor.get());
  outputs.push_back(presentkey_tensor.get());
  outputs.push_back(presentvalue_tensor.get());
                             
  if (!g_gpt3WithoutDecoderOp) {
    AclTransformer::GPT3LayerParam param =
      {epsilon, begin_norm_axis, head_dim, head_num, layer_num};
    g_gpt3WithoutDecoderOp.reset(new AclTransformer::GPT3LayerWithoutCacheDecoderOperation(param));
    g_gpt3WithoutCachePlan.reset(new AclTransformer::Plan);
    g_gpt3WithoutDecoderOp->BuildPlan(g_gpt3WithoutCachePlan.get());
  }

  AclTransformer::VariantPack variantPack;
  BuildVariantPack(inputs, outputs, variantPack);

  /* Set up */
  AsdOps::Status st = g_gpt3WithoutCachePlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External(
                    "GPT3LayerWithoutCacheOp Setup plan failed,"
                    "ret message: %s .", st.Message()));

variantPack.workspaceSize = g_gpt3WithoutCachePlan->GetWorkspaceSize();

  if (variantPack.workspaceSize > 0) {
      SetWorkspace(variantPack.workspaceSize);
      variantPack.workspace = GetWorkspace();
  }
  /* Execute */
  st = g_gpt3WithoutCachePlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External(
                    "GPT3LayerWithoutCacheOp Execute plan failed,"
                    "ret message: %s .", st.Message()));
                    
  static uint64_t executeCount_ = 0;
  executeCount_++;
  if ((executeCount_) % layer_num == 0) { // 1.....32,第32次同步
    int ret = aclrtSynchronizeStream(stream);
  }  

  return {paddle::Tensor(gpt3layerout_tensor),
          paddle::Tensor(presentkey_tensor),
          paddle::Tensor(presentvalue_tensor)};
}

PD_BUILD_OP(gpt3_layer_without_kvcache)
    .Inputs({
      "Hidden",
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
    .SetKernelFn(PD_KERNEL(GPT3LayerWithoutCacheOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        GPT3LayerWithoutCacheOpInferShape));  // neccessary if the op has muti_inputs

#endif
