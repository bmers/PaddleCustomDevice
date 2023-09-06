/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "llamalayer_encoder_parallel_operation.h"

enum LLaMALayerEncoderParallelTensorId
{
  IN_HIDDENSTATES = 0,
  IN_NORMWEIGHT,
  IN_QMIXDWEIGHT,
  IN_KMIXDWEIGHT,
  IN_VMIXDWEIGHT,
  IN_SELFOUTLINEARWEIGHT,
  IN_SELFOUTNORMWEIGHT,
  IN_MLPGATEWEIGHT,
  IN_MLPDOWNWEIGHT,
  IN_MLPUPWEIGHT,
  IN_POSITIONIDS,
  IN_COSTABLE,
  IN_SINTABLE,
  IN_ATTENTIONMASK,
  OUT_LLAMA65BLAYEROUT,
  OUT_PRESENTKEY,
  OUT_PRESENTVALUE,
  INTERMIDATE_INPUTNORMOUT,
  INTERMIDATE_MIXEDQ,
  INTERMIDATE_MIXEDK,
  INTERMIDATE_MIXEDV,
  INTERMIDATE_POSITIONEMBEDQ,
  INTERMIDATE_POSITIONEMBEDK,
  INTERMIDATE_SELFOUT,
  INTERMIDATE_SELFLINEAROUT,
  INTERMIDATE_SELFRESIDUALADDOUT,
  INTERMIDATE_SELFNORMOUT,
  INTERMIDATE_MLPOUT,
  INTERMIDATE_MLPLINEARPARALLELOUT,
};

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 13;

atb::Status CreateLlamaLayerEncoderParallelOperation(const LlamaLayerEncoderParallelParam &param,
                                               atb::Operation **operation)
{
  atb::GraphParam opGraph;
  opGraph.inTensorNum = IN_TENSOR_COUNT;
  opGraph.outTensorNum = OUT_TENSOR_COUNT;
  opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
  opGraph.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
  atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
  atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
  atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
  atb::Node &qPositionEmbeddingNode = opGraph.nodes.at(nodeId++);
  atb::Node &kPositionEmbeddingNode = opGraph.nodes.at(nodeId++);
  atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
  atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
  atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
  atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
  atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
  atb::Node &mlpLinearParallelNode = opGraph.nodes.at(nodeId++);
  atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

  atb::infer_old::RmsNormParam inputNormParam;
  inputNormParam.rmsNormEps = param.rmsNormEps;
  CreateOp(inputNormParam, &inputNormNode.op);
  inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
  inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

  atb::infer::LinearParam mixdQLinearParam;
  mixdQLinearParam.transposeA = false;
  mixdQLinearParam.transposeB = false;
  mixdQLinearParam.hasBias = false;
  CreateOp(mixdQLinearParam, &mixdQLinearNode.op);
  mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT};
  mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

  atb::infer::LinearParam mixdKLinearParam;
  mixdKLinearParam.transposeA = false;
  mixdKLinearParam.transposeB = false;
  mixdKLinearParam.hasBias = false;
  CreateOp(mixdKLinearParam, &mixdKLinearNode.op);
  mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT};
  mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

  atb::infer::LinearParam mixdVLinearParam;
  mixdVLinearParam.transposeA = false;
  mixdVLinearParam.transposeB = false;
  mixdVLinearParam.hasBias = false;
  CreateOp(mixdVLinearParam, &mixdVLinearNode.op);
  mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT};
  mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

  atb::infer_old::PositionEmbedding1dSplitParam positionEmbedding1dSplitQParam;
  positionEmbedding1dSplitQParam.headNum = param.headNum;
  CreateOp(positionEmbedding1dSplitQParam, &qPositionEmbeddingNode.op);
  qPositionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDQ, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
  qPositionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ};

  atb::infer_old::PositionEmbedding1dSplitParam positionEmbedding1dSplitKParam;
  positionEmbedding1dSplitKParam.headNum = param.headNum;
  CreateOp(positionEmbedding1dSplitKParam, &kPositionEmbeddingNode.op);
  kPositionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDK, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
  kPositionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDK};

  atb::infer_old::SelfAttentionParam selfAttentionParam;
  selfAttentionParam.dk = param.dk;
  selfAttentionParam.headNum = param.headNum;
  selfAttentionParam.model = "llama7b";
  CreateOp(selfAttentionParam, &selfAttentionNode.op);
  selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                    INTERMIDATE_POSITIONEMBEDK,
                                    INTERMIDATE_MIXEDV,
                                    IN_ATTENTIONMASK};
  selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};
  selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
  selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    newShape.dimNum = 4; // dimNum: 4
    newShape.dims[0] = oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[0];
    newShape.dims[2] = param.headNum;
    newShape.dims[3] = oldShape.dims[2] / param.headNum;
  };

  atb::infer_old::LinearParallelParam selfOutLinearParallelParam;
  selfOutLinearParallelParam.transWeight = false;
  selfOutLinearParallelParam.rank = param.rank;
  selfOutLinearParallelParam.rankSize = param.rankSize;
  selfOutLinearParallelParam.rankRoot = 0;
  selfOutLinearParallelParam.bias = "None";
  selfOutLinearParallelParam.parallelType = "RowParallel";
  selfOutLinearParallelParam.backend = "hccl";
  selfOutLinearParallelParam.useCommExt = param.useCommExt;
  selfOutLinearParallelParam.commExt = param.commExt;
  CreateOp(selfOutLinearParallelParam, &selfOutLinearParallelNode.op);
  selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
  selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

  atb::infer_old::AddParam selfResidualAddParam;
  CreateOp(selfResidualAddParam, &selfResidualAddNode.op);
  selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
  selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
  selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
  selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    newShape.dimNum = 3; // dimNum: 3
    newShape.dims[0] = oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[0];
    newShape.dims[2] = oldShape.dims[2];
  };

  atb::infer_old::RmsNormParam selfNormParam;
  selfNormParam.rmsNormEps = param.rmsNormEps;
  CreateOp(selfNormParam, &selfNormNode.op);
  selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
  selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

  atb::infer_old::MlpParam mlpParam;
  mlpParam.model = "llama13b";
  CreateOp(mlpParam, &mlpNode.op);
  mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPUPWEIGHT};
  mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

  atb::infer_old::LinearParallelParam mlpLinearParallelParam;
  mlpLinearParallelParam.transWeight = false;
  mlpLinearParallelParam.rank = param.rank;
  mlpLinearParallelParam.rankSize = param.rankSize;
  mlpLinearParallelParam.rankRoot = 0;
  mlpLinearParallelParam.bias = "None";
  mlpLinearParallelParam.parallelType = "RowParallel";
  mlpLinearParallelParam.backend = "hccl";
  mlpLinearParallelParam.useCommExt = param.useCommExt;
  mlpLinearParallelParam.commExt = param.commExt;
  CreateOp(mlpLinearParallelParam, &mlpLinearParallelNode.op);
  mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPDOWNWEIGHT};
  mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT};

  atb::infer_old::AddParam mlpResidualAddParam;
  CreateOp(mlpResidualAddParam, &mlpResidualAddNode.op);
  mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEARPARALLELOUT};
  mlpResidualAddNode.outTensorIds = {OUT_LLAMA65BLAYEROUT};

  opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).shape.dimNum = 4;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(1).shape.dims[2] = param.headNum;
    outTensorDescs.at(1).shape.dims[3] = param.dk;
    outTensorDescs.at(2) = outTensorDescs.at(1);
    return atb::NO_ERROR;
  };

  atb::CreateOp(opGraph, operation);
  return atb::NO_ERROR;
}
