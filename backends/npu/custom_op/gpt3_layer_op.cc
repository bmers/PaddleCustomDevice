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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "gpt3_layer_op.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "kernels/funcs/format_utils.h"
#include "acltransformer/params/add.h"
#endif

std::vector<std::vector<int64_t>> AddOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape) {
  return {x_shape};
}


// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC

struct GPT3LayerParam {
    double layerNormEps = 0;
    int headNum = 0;
    bool transKey = false;
    int dk = 0;
    int layerId = 0;
    float residualAddScale = 0;
};

std::unique_ptr<GPT3LayerDecoderOperation> g_gpt3DecoderOp;
std::unique_ptr<AclTransformer::Plan> g_gpt3Plan;

staic auto layer = (parm);

enum GPT3LayerDecoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_SEQLEN,
    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_FFNLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

GPT3LayerDecoderOperation::GPT3LayerDecoderOperation(const GPT3LayerParam &param)
    : GraphOperation("GPT3LayerDecoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(new AclTransformer::RopeOperation({param_.headNum}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE,
                                         IN_SEQLEN};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    selfAttentionKvCacheNode.operation.reset(new AclTransformer::SelfAttentionKvCacheOperation(
        {param_.transKey, param_.dk, param_.headNum, param_.layerId}));
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_VALUE,
                                            IN_ATTENTIONMASK,
                                            IN_PASTKEY,
                                            IN_PASTVALUE};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    ffnNode.operation.reset(new AclTransformer::FfnOperation({}));
    ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
    ffnNode.outTensorIds = {INTERMIDATE_FFNOUT};

    ffnLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT};

    ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_FFNLINEAROUT};
    ffnResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};
}

GPT3LayerDecoderOperation::~GPT3LayerDecoderOperation() {}

uint64_t GPT3LayerDecoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GPT3LayerDecoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GPT3LayerDecoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_PASTKEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_PASTVALUE);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(0) += 1;
    const size_t tensorId2 = 2;
    outTensorDescs.at(tensorId2) = valueTensor.desc;
    outTensorDescs.at(tensorId2).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}
// #endif

void OperationTorch::BuildVariantPack(std::vector<phi::DenseTensor> &inTensors, std::vector<phi::DenseTensor> &outTensors,
                                      AclTransformer::VariantPack &variantPack)
{
    variantPack.inTensors.resize(inTensors.size());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " execute start, inTensors[" << i << "].options:" << inTensors.at(i).options()
                      << ", data:" << inTensors.at(i).data_ptr()
                      << ", storage_offset:" << inTensors.at(i).storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(inTensors.at(i));
        variantPack.inTensors.at(i) = ConvertDenseTensorToAsdTensor(inTensors.at(i));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
            variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + opName_ + "/intensor" + std::to_string(i) + ".pth";

            ASD_LOG(INFO) << operation_->GetName() << " save tensor:" << filePath;
        }
    }

    variantPack.outTensors.resize(outTensors.size());
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " execute start, outTensors[" << i << "].options:" << outTensors.at(i).options()
                      << ", data:" << outTensors.at(i).data_ptr()
                      << ", storage_offset:" << outTensors.at(i).storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(outTensors.at(i));
        variantPack.outTensors.at(i) = ConvertDenseTensorToAsdTensor(outTensors.at(i));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.outTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
            variantPack.outTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + opName_ + "/outtensor" + std::to_string(i) + ".pth";
            ASD_LOG(INFO) << operation_->GetName() << " save tensor:" << filePath;
        }
    }
}

std::vector<paddle::Tensor> GPT3LayerOp(
    const paddle::Tensor& x_tensor,
    const paddle::Tensor& y_tensor) {

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x_tensor.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  
  auto x = static_cast<const phi::DenseTensor*>(x_tensor.impl().get());
  auto y = static_cast<const phi::DenseTensor*>(y_tensor.impl().get());

  if (!g_gpt3DecoderOp) {
    GPT3LayerParam param;
    g_gpt3DecoderOp.reset(new GPT3LayerDecoderOperation(param));
    g_gpt3Plan.reset(new AclTransformer::Plan);
    g_decodeOp->BuildPlan(g_gpt3Plan.get());
  }

  AclTransformer::VariantPack variantPack;

  g_gpt3Plan->Setup(dev_ctx, variantPack)
  operation_

#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC

#endif
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AddOpInferShape));  // neccessary if the op has muti_inputs
