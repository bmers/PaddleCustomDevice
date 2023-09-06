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
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/time/timer.h>

#include <iostream>
#include <vector>

#include "acltransformer/config.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"
#include "gpt3_layer_op.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#include "runtime/runtime.h"
#include "self_attention_kv_cache_fusion_gpt3/self_attention_kv_cache_fusion_gpt3_operation.h"

namespace AclTransformer {
enum GPT3LayerDecoderParallelTensorId {
  IN_HIDDENSTATES_PARALLEL = 0,
  IN_NORMWEIGHT_PARALLEL,
  IN_NORMBIAS_PARALLEL,
  IN_QKVMIXDWEIGHT_PARALLEL,
  IN_QKVMIXDBIAS_PARALLEL,
  IN_SELFOUTLINEARWEIGHT_PARALLEL,
  IN_SELFOUTLINEARBIAS_PARALLEL,
  IN_SELFOUTNORMWEIGHT_PARALLEL,
  IN_SELFOUTNORMBIAS_PARALLEL,
  IN_FFNLINEARWEIGHT_PARALLEL,
  IN_FFNLINEARBIAS_PARALLEL,
  IN_FFNOUTLINEARWEIGHT_PARALLEL,
  IN_FFNOUTLINEARBIAS_PARALLEL,
  IN_SEQLEN_PARALLEL,
  IN_TOKENOFFSET_PARALLEL,
  IN_LAYERID_PARALLEL,
  IN_ATTENTIONMASK_PARALLEL,
  IN_PASTKEY_PARALLEL,
  IN_PASTVALUE_PARALLEL,

  OUT_GPT3LAYEROUT_PARALLEL,

  INTER_INPUTNORMOUT_PARALLEL,
  INTER_MIXEDLINEAROUTQKV_PARALLEL,

  INTER_SELFOUT_PARALLEL,
  INTER_SELFLINEAROUT_PARALLEL,
  INTER_SELFRESIDUALADDOUT_PARALLEL,
  INTER_SELFNORMOUT_PARALLEL,
  INTER_FFNOUT_PARALLEL,
  INTER_FFNLINEAROUT_PARALLEL,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

GPT3LayerDecoderParallelOperation::GPT3LayerDecoderParallelOperation(
    const GPT3LayerParam &param)
    : GraphOperation("GPT3LayerDecoderParallelOperation"), param_(param) {
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

  inputNormNode.operation.reset(
      new AclTransformer::NormOperation({param_.layerNormEps,
                                         param_.layerNormBeginNormAxis,
                                         param_.layerNormBeginNormAxis}));
  inputNormNode.inTensorIds = {
      IN_HIDDENSTATES_PARALLEL, IN_NORMWEIGHT_PARALLEL, IN_NORMBIAS_PARALLEL};
  inputNormNode.outTensorIds = {INTER_INPUTNORMOUT_PARALLEL};

  mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation(
      {false, true})); /* 加速库默认会将w进行转置 */
  mixdQkvLinearNode.inTensorIds = {INTER_INPUTNORMOUT_PARALLEL,
                                   IN_QKVMIXDWEIGHT_PARALLEL,
                                   IN_QKVMIXDBIAS_PARALLEL};
  mixdQkvLinearNode.outTensorIds = {INTER_MIXEDLINEAROUTQKV_PARALLEL};

  selfAttentionKvCacheNode.operation.reset(
      new AclTransformer::SelfAttentionKvCacheFusionGPT3Operation(
          {param_.head_num,
           param_.layer_num - 1,
           param_.head_dim,
           0,
           0,
           0,
           "chatglm6b",
           param_.seqLen,
           param_.tokenOffset}));
  selfAttentionKvCacheNode.inTensorIds = {INTER_MIXEDLINEAROUTQKV_PARALLEL,
                                          IN_PASTKEY_PARALLEL,
                                          IN_PASTVALUE_PARALLEL,
                                          IN_ATTENTIONMASK_PARALLEL,
                                          IN_TOKENOFFSET_PARALLEL,
                                          IN_SEQLEN_PARALLEL,
                                          IN_LAYERID_PARALLEL};
  selfAttentionKvCacheNode.outTensorIds = {INTER_SELFOUT_PARALLEL};
  selfAttentionKvCacheNode.useVariantPackParam = true;

  selfOutLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
      {true, 0, 0, 0, "YES", "RowParallel", "hccl", true, param_.comm}));
  selfOutLinearNode.inTensorIds = {INTER_SELFOUT_PARALLEL,
                                   IN_SELFOUTLINEARWEIGHT_PARALLEL,
                                   IN_SELFOUTLINEARBIAS_PARALLEL};
  selfOutLinearNode.outTensorIds = {INTER_SELFLINEAROUT_PARALLEL};

  selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES_PARALLEL,
                                     INTER_SELFLINEAROUT_PARALLEL};
  selfResidualAddNode.outTensorIds = {INTER_SELFRESIDUALADDOUT_PARALLEL};

  selfNormNode.operation.reset(
      new AclTransformer::NormOperation({param_.layerNormEps}));
  selfNormNode.inTensorIds = {INTER_SELFRESIDUALADDOUT_PARALLEL,
                              IN_SELFOUTNORMWEIGHT_PARALLEL,
                              IN_SELFOUTNORMBIAS_PARALLEL};
  selfNormNode.outTensorIds = {INTER_SELFNORMOUT_PARALLEL};

  ffnNode.operation.reset(new AclTransformer::FfnOperation({false, true}));
  ffnNode.inTensorIds = {INTER_SELFNORMOUT_PARALLEL,
                         IN_FFNLINEARWEIGHT_PARALLEL,
                         IN_FFNLINEARBIAS_PARALLEL};
  ffnNode.outTensorIds = {INTER_FFNOUT_PARALLEL};

  ffnLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
      {true, 0, 0, 0, "YES", "RowParallel", "hccl", true, param_.comm}));
  ffnLinearNode.inTensorIds = {INTER_FFNOUT_PARALLEL,
                               IN_FFNOUTLINEARWEIGHT_PARALLEL,
                               IN_FFNOUTLINEARBIAS_PARALLEL};
  ffnLinearNode.outTensorIds = {INTER_FFNLINEAROUT_PARALLEL};

  ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  ffnResidualAddNode.inTensorIds = {INTER_SELFRESIDUALADDOUT_PARALLEL,
                                    INTER_FFNLINEAROUT_PARALLEL};
  ffnResidualAddNode.outTensorIds = {OUT_GPT3LAYEROUT_PARALLEL};
}

GPT3LayerDecoderParallelOperation::~GPT3LayerDecoderParallelOperation() {}

uint64_t GPT3LayerDecoderParallelOperation::GetInTensorCount() const {
  return IN_TENSOR_COUNT;
}

uint64_t GPT3LayerDecoderParallelOperation::GetOutTensorCount() const {
  return OUT_TENSOR_COUNT;
}

AsdOps::Status GPT3LayerDecoderParallelOperation::InferShapeImpl(
    const AsdOps::SVector<AsdOps::Tensor> &inTensors,
    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const {
  outTensorDescs.at(0) = inTensors.at(0).desc;

  return AsdOps::Status::OkStatus();
}

}  // namespace AclTransformer

GPT3LayerWorkspace g_gpt3ParallelWorkSpace = {nullptr, 0};

std::map<HcclComm,
         std::pair<
             std::shared_ptr<AclTransformer::GPT3LayerDecoderParallelOperation>,
             std::shared_ptr<AclTransformer::Plan>>>
    g_gpt3DecoderMap;

static AclTransformer::SelfAttentionKvCacheFusionVariantPackParam
    g_variantPackParam_;
static AsdOps::Tensor g_gpt3_cachek;
static AsdOps::Tensor g_gpt3_cachev;
static AsdOps::Tensor g_gpt3_attenmask;

// static std::vector<int32_t> g_seq_len_vector(1, 1); /* 增量的q_seq_len，为1，当前也只考虑batch为1 */
// static AsdOps::SVector<int32_t> g_seq_len(1, 1);
static std::vector<int32_t> g_token_offset_vector;
static AsdOps::SVector<int32_t> g_token_offset;

std::vector<std::vector<int64_t>> GPT3LayerParallelOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &norm_bias_shape,
    const std::vector<int64_t> &mix_linear_weight_shape,
    const std::vector<int64_t> &mix_linear_bias_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_linear_bias_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &self_out_norm_bias_shape,
    const std::vector<int64_t> &ffn_linear_weight_shape,
    const std::vector<int64_t> &ffn_linear_bias_shape,
    const std::vector<int64_t> &ffn_out_linear_weight_shape,
    const std::vector<int64_t> &ffn_out_linear_bias_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &pastkey_shape,
    const std::vector<int64_t> &pastvalue_shape) {
  return {hidden_shape, pastkey_shape, pastvalue_shape};
}

static void BuildVariantPack(
    AsdOps::SVector<const phi::DenseTensor *> &inTensors,
    phi::DenseTensor &outTensors,
    AclTransformer::VariantPack &variantPack) {
  variantPack.inTensors.resize(inTensors.size() + 3);
  for (size_t i = 0; i < inTensors.size(); ++i) {
    variantPack.inTensors.at(i) =
        ConvertDenseTensorToAsdTensor(*(inTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }
  variantPack.inTensors.at(inTensors.size()) = g_gpt3_attenmask;
  variantPack.inTensors.at(inTensors.size() + 1) = g_gpt3_cachek;
  variantPack.inTensors.at(inTensors.size() + 2) = g_gpt3_cachev;

  variantPack.outTensors.resize(1);
  variantPack.outTensors.at(0) = ConvertDenseTensorToAsdTensor(outTensors);
  if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
      variantPack.outTensors.at(0).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
    variantPack.outTensors.at(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
  }
}

static void SetWorkspace(uint64_t workspaceSize) {
  if (workspaceSize <= g_gpt3ParallelWorkSpace.workspaceSize_) {
    VLOG(4) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_gpt3ParallelWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_gpt3ParallelWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_gpt3ParallelWorkSpace.workspace_);
    g_gpt3ParallelWorkSpace.workspace_ = nullptr;
    g_gpt3ParallelWorkSpace.workspaceSize_ = 0;
  }

  int st = AsdRtMemMallocDevice((void **)&(g_gpt3ParallelWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);

  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("GPT3LayerParallelOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_gpt3ParallelWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_gpt3ParallelWorkSpace.workspace_; }

void GPT3LayerParallelGetTensorInputs(
    const phi::CustomContext &dev_ctx,
    uint32_t layer_id,
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_bias,
    const paddle::Tensor &mix_linear_weight,
    const paddle::Tensor &mix_linear_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_bias,
    const paddle::Tensor &ffn_linear_weight,
    const paddle::Tensor &ffn_linear_bias,
    const paddle::Tensor &ffn_out_linear_weight,
    const paddle::Tensor &ffn_out_linear_bias,
    phi::DenseTensor &seq_len_dense,
    phi::DenseTensor &token_offset_dense,
    phi::DenseTensor &layer_id_dense,
    AsdOps::SVector<const phi::DenseTensor *> &inputs) {
  auto hidden_tensor =
      static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor =
      static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_bias_tensor =
      static_cast<const phi::DenseTensor *>(norm_bias.impl().get());
  auto mix_linear_weight_tensor =
      static_cast<const phi::DenseTensor *>(mix_linear_weight.impl().get());
  auto mix_linear_bias_tensor =
      static_cast<const phi::DenseTensor *>(mix_linear_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<const phi::DenseTensor *>(
      self_out_linear_weight.impl().get());
  auto self_out_linear_bias_tensor =
      static_cast<const phi::DenseTensor *>(self_out_linear_bias.impl().get());
  auto self_out_norm_weight_tensor =
      static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_bias_tensor =
      static_cast<const phi::DenseTensor *>(self_out_norm_bias.impl().get());
  auto ffn_linear_weight_tensor =
      static_cast<const phi::DenseTensor *>(ffn_linear_weight.impl().get());
  auto ffn_linear_bias_tensor =
      static_cast<const phi::DenseTensor *>(ffn_linear_bias.impl().get());
  auto ffn_out_linear_weight_tensor =
      static_cast<const phi::DenseTensor *>(ffn_out_linear_weight.impl().get());
  auto ffn_out_linear_bias_tensor =
      static_cast<const phi::DenseTensor *>(ffn_out_linear_bias.impl().get());

  std::vector<int32_t> layer_id_vec(1, layer_id);
  std::vector<int32_t> g_seq_len_vector(hidden.shape().at(0), 1);
  
  custom_kernel::TensorFromVector(
      dev_ctx, g_seq_len_vector, dev_ctx, &seq_len_dense);
  custom_kernel::TensorFromVector(
      dev_ctx, g_token_offset_vector, dev_ctx, &token_offset_dense);
  custom_kernel::TensorFromVector(
      dev_ctx, layer_id_vec, dev_ctx, &layer_id_dense);

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
  inputs.push_back(&seq_len_dense);
  inputs.push_back(&token_offset_dense);
  inputs.push_back(&layer_id_dense);
}

static void InitFlashAttentionTensor(int layer_num,
                              int batch,
                              int org_seq_len,
                              int max_seqlen,
                              int head_dim,
                              int head_num,
                              const phi::DenseTensor *cache_k,
                              const phi::DenseTensor *cache_v) {
  /* [layer_num, batch, max_seqlen, hidden_size] */
  uint64_t cache_k_size = batch * max_seqlen * head_dim * head_num * layer_num *
                          sizeof(std::uint16_t); /* 应该是float16 */

  g_gpt3_cachek.desc = {AsdOps::TENSOR_DTYPE_FLOAT16,
                        AsdOps::TENSOR_FORMAT_ND,
                        {layer_num, batch, max_seqlen, head_dim * head_num}};
  g_gpt3_cachek.dataSize = cache_k_size;
  int st = AsdRtMemMallocDevice(
      (void **)&(g_gpt3_cachek.data), cache_k_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("Alloc g_gpt3_cachek AsdRtMemMallocDevice,"
                            "fail, ret: %d size: %d.",
                            st,
                            cache_k_size));
  AsdRtMemCopy(g_gpt3_cachek.data,
               cache_k_size,
               const_cast<void *>(cache_k->data()),
               cache_k_size,
               ASDRT_MEMCOPY_HOST_TO_DEVICE);

  g_gpt3_cachev.desc = {AsdOps::TENSOR_DTYPE_FLOAT16,
                        AsdOps::TENSOR_FORMAT_ND,
                        {layer_num, batch, max_seqlen, head_dim * head_num}};
  g_gpt3_cachev.dataSize = cache_k_size;
  st = AsdRtMemMallocDevice(
      (void **)&(g_gpt3_cachev.data), cache_k_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("Alloc g_gpt3_cachev AsdRtMemMallocDevice,"
                            "fail, ret: %d size: %d.",
                            st,
                            cache_k_size));
  AsdRtMemCopy(g_gpt3_cachev.data,
               cache_k_size,
               const_cast<void *>(cache_v->data()),
               cache_k_size,
               ASDRT_MEMCOPY_HOST_TO_DEVICE);

  uint64_t atten_size = max_seqlen * max_seqlen * sizeof(std::uint16_t);
  g_gpt3_attenmask.desc = {AsdOps::TENSOR_DTYPE_FLOAT16,
                           AsdOps::TENSOR_FORMAT_ND,
                           {max_seqlen, max_seqlen}};
  g_gpt3_attenmask.dataSize = atten_size;
  st = AsdRtMemMallocDevice(
      (void **)&(g_gpt3_attenmask.data), atten_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("Alloc g_gpt3_attenmask AsdRtMemMallocDevice,"
                            "fail, ret: %d size: %d.",
                            st,
                            atten_size));
}

std::vector<paddle::Tensor> GPT3LayerParallelOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_bias,
    const paddle::Tensor &mix_linear_weight,
    const paddle::Tensor &mix_linear_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_bias,
    const paddle::Tensor &ffn_linear_weight,
    const paddle::Tensor &ffn_linear_bias,
    const paddle::Tensor &ffn_out_linear_weight,
    const paddle::Tensor &ffn_out_linear_bias,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &past_key,
    const paddle::Tensor &past_value,
    int begin_norm_axis,
    float epsilon,
    std::vector<int32_t> shape,
    float scale) {

  AsdOps::Timer timer;
  int32_t batch_size = past_key.shape().at(0);
  int32_t org_seq_len = past_key.shape().at(1);
  int32_t layer_num = (int32_t)scale;
  int32_t head_dim = shape[3] / 3;
  int32_t head_num = self_out_linear_weight.shape()[0] / head_dim;

  static uint32_t layerId = 0;

  if (layerId % layer_num == 0) { /* 0 ....31,第2个token32次时候清零，调整 */
    layerId = 0;
    if (g_gpt3DecoderMap.empty()) { /* token_offset为kvLen，第一个token，初始化为org_seq_len
                                     */

      //int org_seq_len = past_key.shape().at(1);
      //int batch_tmp = 1;
      g_token_offset_vector.clear();
      g_token_offset_vector.resize(batch_size, org_seq_len);
	  AsdOps::SVector<int32_t> g_token_offset_t(batch_size, org_seq_len);
      g_token_offset = g_token_offset_t;
      // g_token_offset.clear();
      // g_token_offset.resize(batch_tmp);
      // g_token_offset.push_back(org_seq_len);
    } else { /* 每处理个token，长度加1 */
      std::for_each(g_token_offset_vector.begin(),
                    g_token_offset_vector.end(),
                    [](int &n) { n++; });
      std::for_each(
          g_token_offset.begin(), g_token_offset.end(), [](int &n) { n++; });
    }
  }

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};
  HcclComm comm = reinterpret_cast<HcclComm>(dev_ctx->xccl_comm());

  phi::DenseTensor seq_len_tensor;
  phi::DenseTensor token_offset_tensor;
  phi::DenseTensor layer_id_tensor;
  AsdOps::SVector<const phi::DenseTensor *> inputs;

  GPT3LayerParallelGetTensorInputs(*dev_ctx,
                                   layerId,
                                   hidden,
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
                                   seq_len_tensor,
                                   token_offset_tensor,
                                   layer_id_tensor,
                                   inputs);
  layerId++;

  std::shared_ptr<phi::DenseTensor> gpt3layerout_tensor =
      std::make_shared<phi::DenseTensor>();
  gpt3layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(gpt3layerout_tensor.get(), inputs.at(0)->dtype());
  std::shared_ptr<AclTransformer::GPT3LayerDecoderParallelOperation>
      gpt3decoderOp;
  std::shared_ptr<AclTransformer::Plan> gpt3plan;

  AsdOps::SVector<int32_t> g_seq_len(batch_size, 1);
  
  auto it = g_gpt3DecoderMap.find(comm);
  if (it == g_gpt3DecoderMap.end()) {
    std::cout << "GPT3LayerParallelOp comm: " << comm << std::endl;
    std::shared_ptr<AclTransformer::GPT3LayerDecoderParallelOperation>
        decoderOp;
    std::shared_ptr<AclTransformer::Plan> plan;

    InitFlashAttentionTensor(layer_num,
                             batch_size,
                             org_seq_len,
                             GPT3_LAYER_FLASH_ATTENTION_MAX_SEQ_LEN,
                             head_dim,
                             head_num,
                             inputs[inputs.size() - 2],
                             inputs[inputs.size() - 1]);
    AclTransformer::GPT3LayerParam param = {epsilon,
                                            begin_norm_axis,
                                            head_dim,
                                            head_num,
                                            layer_num,
                                            g_token_offset,
                                            g_seq_len,
                                            comm};
    decoderOp.reset(
        new AclTransformer::GPT3LayerDecoderParallelOperation(param));
    plan.reset(new AclTransformer::Plan);
    decoderOp->BuildPlan(plan.get());
    g_gpt3DecoderMap[comm] = std::make_pair(decoderOp, plan);
    gpt3decoderOp = decoderOp;
    gpt3plan = plan;
  } else {
    gpt3decoderOp = it->second.first;
    gpt3plan = it->second.second;
  }

  static uint64_t executeCount_ = 0;

  AclTransformer::VariantPack variantPack;
  AclTransformer::SelfAttentionKvCacheFusionVariantPackParam variantPackParam = {g_seq_len, g_token_offset};

  variantPack.param = variantPackParam;
  BuildVariantPack(inputs, *gpt3layerout_tensor.get(), variantPack);

  /* Set up */
  AsdOps::Timer timer1;
  AsdOps::Status st = gpt3plan->Setup(handle, variantPack);

  AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime +=
      timer1.ElapsedMicroSecond();
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("GPT3LayerParallelOp Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));

  variantPack.workspaceSize = gpt3plan->GetWorkspaceSize();
  VLOG(4) << " GPT3LayerParallelOp plan workspace size:" << variantPack.workspaceSize;

  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }

  /* Execute */
  AsdOps::Timer timer2;
  st = gpt3plan->Execute(handle, variantPack);

  AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime +=
      timer2.ElapsedMicroSecond();
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("GPT3LayerParallelOp Execute plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime +=
      timer.ElapsedMicroSecond();
  ASD_LOG(FATAL) << GPT3LayerParallelOp << " executeCount:" << executeCount_++
                 << ", statistic:["
                 << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString()
                 << "]";
  AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();
  if ((executeCount_) % layer_num == 0) {  // 1.....32,第32次同步
    int ret = aclrtSynchronizeStream(stream);
  }
  return {paddle::Tensor(gpt3layerout_tensor), past_key, past_value};
}

GPT3LayerParallelCustomOp::GPT3LayerParallelCustomOp(int layerNum, int batchSize, AclTransformer::Handle handle)
{
  /* Init Task Queue */
  std::thread thread = std::thread(std::bind(&GPT3LayerParallelCustomOp::ThreadProcessTask, this));
  taskProcessThread_ = std::move(thread);

  std::string device_id_str = getenv("FLAGS_selected_npus");
  currentDevId_ = stoi(device_id_str);

  handle_ = handle;
  layerNum_ = layerNum;
  curBatchSize_ = batchSize;
  layerCount_ = 0;
  allTaskFinish_ = false;
}

void GPT3LayerParallelCustomOp::SetParam(AclTransformer::GPT3LayerParam &param)
{
  variantPacks_.resize(layerNum_);
  operations_.resize(layerNum_);
  plans_.resize(layerNum_);

  for (int i = 0; i < layerNum_; i++) {
    /* TODO:修改为模板，即可new不同对象 */
    AclTransformer::GraphOperation *op = new AclTransformer::GPT3LayerDecoderParallelOperation(param);

    operations_.at(i).reset(op);
    AclTransformer::Plan *plan = new AclTransformer::Plan();
    op->BuildPlan(plan);
    plans_.at(i).reset(plan);
  }
}

void GPT3LayerParallelCustomOp::PushTask(int layerId)
{
  std::unique_lock<std::mutex> lock(mutex_);
  taskQueue_.push(layerId);
  lock.unlock();
  cond_.notify_one();
}

int GPT3LayerParallelCustomOp::PopTask()
{
  std::unique_lock<std::mutex> lock(mutex_);
  while (taskQueue_.empty()) {
      cond_.wait(lock);
  }
  int layerId = taskQueue_.front();
  taskQueue_.pop();
  return layerId;
}

void GPT3LayerParallelCustomOp::ExecutePlan(int layerId)
{
  AclTransformer::Plan &plan = *plans_.at(layerId);
  AclTransformer::VariantPack &variantPack = variantPacks_.at(layerId);

  AsdOps::Status st = plan.Execute(handle_, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("GPT3LayerParallelOp %dth Execute plan failed,"
                                          "ret message: %s .", layerId,
                                          st.Message()));
}

void GPT3LayerParallelCustomOp::ThreadProcessTask()
{
  ASD_LOG(FATAL) << "GPT3Layer ThreadProcessTask start";

  int ret = AsdRtDeviceSetCurrent(currentDevId_);
  ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

  int processTaskCount = 0;
  while (true) {
      int layerId = PopTask();
      ExecutePlan(layerId);
      processTaskCount++;
      if (processTaskCount == layerNum_) {
          ASD_LOG(INFO) << "GPT3Layer thread process all layers";
          processTaskCount = 0;
          allTaskFinish_ = true;
      }
  }
}

void GPT3LayerParallelCustomOp::SyncFinalLayer(aclrtStream stream)
{
  if (layerCount_ != layerNum_) return; /* 非最后一层layer, 则直接返回 */

  while (!allTaskFinish_) {}; /* 等待最后一个layer execute */
  aclError ret = aclrtSynchronizeStream(stream);
  PADDLE_ENFORCE_NPU_SUCCESS(ret);

  layerCount_ = 0;
  allTaskFinish_ = false;
}

int GPT3LayerParallelCustomOp::GetCurBatchSize()
{
  return curBatchSize_;
}

AclTransformer::VariantPack& GPT3LayerParallelCustomOp::GetVariantPack(int layerId)
{
  return variantPacks_.at(layerId);
}

void GPT3LayerParallelCustomOp::Setup(int layerId, AclTransformer::Handle handle)
{
  AclTransformer::Plan &plan = *plans_.at(layerId);
  AclTransformer::VariantPack &variantPack = variantPacks_.at(layerId);

  AsdOps::Status st = plan.Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("GPT3LayerParallelOp Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));

  variantPack.workspaceSize = plan.GetWorkspaceSize();
  VLOG(4) << " GPT3LayerParallelOp plan workspace size:" << variantPack.workspaceSize;

  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }

  PushTask(layerId);
  layerCount_++;
}

std::shared_ptr<GPT3LayerParallelCustomOp> g_gpt3DecoderParallelCustomOp;
static int g_parallelLayerId = 0;

std::vector<paddle::Tensor> GPT3LayerParallelOpAsync(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_bias,
    const paddle::Tensor &mix_linear_weight,
    const paddle::Tensor &mix_linear_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_bias,
    const paddle::Tensor &ffn_linear_weight,
    const paddle::Tensor &ffn_linear_bias,
    const paddle::Tensor &ffn_out_linear_weight,
    const paddle::Tensor &ffn_out_linear_bias,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &past_key,
    const paddle::Tensor &past_value,
    int begin_norm_axis,
    float epsilon,
    std::vector<int32_t> shape,
    float scale) {

  int32_t batch_size = past_key.shape().at(0);
  int32_t org_seq_len = past_key.shape().at(1);
  int32_t layer_num = (int32_t)scale;
  int32_t head_dim = shape[3] / 3;
  int32_t head_num = self_out_linear_weight.shape()[0] / head_dim;

  if (g_parallelLayerId % layer_num == 0) { /* 0 ....31,第2个token32次时候清零，调整 */
    g_parallelLayerId = 0;
    /* token_offset为kvLen，第一个token，初始化为org_seq_len */
    if (!g_gpt3DecoderParallelCustomOp ||
        g_gpt3DecoderParallelCustomOp->GetCurBatchSize() != batch_size) {
          g_token_offset_vector.clear();
          g_token_offset_vector.resize(batch_size, org_seq_len);
          AsdOps::SVector<int32_t> token_offset_t(batch_size, org_seq_len);
          g_token_offset = token_offset_t;
    } else { /* 每处理个token，长度加1 */
          std::for_each(g_token_offset_vector.begin(),
                        g_token_offset_vector.end(),
                        [](int &n) { n++; });
          std::for_each(g_token_offset.begin(),
                        g_token_offset.end(),
                        [](int &n) { n++; });
    }
  }

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};
  HcclComm comm = reinterpret_cast<HcclComm>(dev_ctx->xccl_comm());

  phi::DenseTensor seq_len_tensor;
  phi::DenseTensor token_offset_tensor;
  phi::DenseTensor layer_id_tensor;
  AsdOps::SVector<const phi::DenseTensor *> inputs;
  AsdOps::SVector<int32_t> q_seq_len(batch_size, 1); /* 增量阶段，q_seq_len始终为1 */

  GPT3LayerParallelGetTensorInputs(*dev_ctx,
                                   g_parallelLayerId,
                                   hidden,
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
                                   seq_len_tensor,
                                   token_offset_tensor,
                                   layer_id_tensor,
                                   inputs);

  std::shared_ptr<phi::DenseTensor> gpt3layerout_tensor =
      std::make_shared<phi::DenseTensor>();
  gpt3layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(gpt3layerout_tensor.get(), inputs.at(0)->dtype());

  if (!g_gpt3DecoderParallelCustomOp || (g_gpt3DecoderParallelCustomOp->GetCurBatchSize() != batch_size)) {
    std::cout << "GPT3LayerParallelOpAsync comm: " << comm << std::endl;
    g_gpt3DecoderParallelCustomOp.reset(new GPT3LayerParallelCustomOp(layer_num, batch_size, handle));

    InitFlashAttentionTensor(layer_num,
                             batch_size,
                             org_seq_len,
                             GPT3_LAYER_FLASH_ATTENTION_MAX_SEQ_LEN,
                             head_dim,
                             head_num,
                             inputs[inputs.size() - 2],
                             inputs[inputs.size() - 1]);

    AclTransformer::GPT3LayerParam param = {epsilon,
                                            begin_norm_axis,
                                            head_dim,
                                            head_num,
                                            layer_num,
                                            g_token_offset,
                                            q_seq_len,
                                            comm};

    g_gpt3DecoderParallelCustomOp->SetParam(param);
  }

  AclTransformer::VariantPack &variantPack = g_gpt3DecoderParallelCustomOp->GetVariantPack(g_parallelLayerId);
  AclTransformer::SelfAttentionKvCacheFusionVariantPackParam variantPackParam = {q_seq_len, g_token_offset};

  variantPack.param = variantPackParam;
  BuildVariantPack(inputs, *gpt3layerout_tensor.get(), variantPack);

  /* Set up */
  g_gpt3DecoderParallelCustomOp->Setup(g_parallelLayerId, handle);
  g_gpt3DecoderParallelCustomOp->SyncFinalLayer(stream);
  g_parallelLayerId++;

  return {paddle::Tensor(gpt3layerout_tensor), past_key, past_value};
}

PD_BUILD_OP(gpt3_layer_parallel)
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
             "AttentionMask",
             "PastKey",
             "PastValue"})
    .Outputs({"Out", "PresentKey", "PresentValue"})
    .Attrs({"begin_norm_axis: int",
            "epsilon: float",
            "shape: std::vector<int>",
            "scale: float"})
    .SetKernelFn(PD_KERNEL(GPT3LayerParallelOp))
    .SetInferShapeFn(
        PD_INFER_SHAPE(GPT3LayerParallelOpInferShape));  // neccessary if the op
                                                         // has muti_inputs

PD_BUILD_OP(gpt3_layer_parallel_async)
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
             "AttentionMask",
             "PastKey",
             "PastValue"})
    .Outputs({"Out", "PresentKey", "PresentValue"})
    .Attrs({"begin_norm_axis: int",
            "epsilon: float",
            "shape: std::vector<int>",
            "scale: float"})
    .SetKernelFn(PD_KERNEL(GPT3LayerParallelOpAsync))
    .SetInferShapeFn(
        PD_INFER_SHAPE(GPT3LayerParallelOpInferShape));  // neccessary if the op
                                                         // has muti_inputs
#endif
