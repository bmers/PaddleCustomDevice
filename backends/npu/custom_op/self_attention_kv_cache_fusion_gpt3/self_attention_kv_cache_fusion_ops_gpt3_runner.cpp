/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include <iostream>
#include "self_attention_kv_cache_fusion_ops_gpt3_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheFusionOpsGPT3Runner::SelfAttentionKvCacheFusionOpsGPT3Runner(
    const SelfAttentionKvCacheFusionParam &param)
    : OpsRunner("SelfAttentionKvCacheFusionOpsGPT3Runner", RUNNER_TYPE_SELF_ATTENTION_KV_FUSION_CACHE),
      param_(param)
{
  auto batch_size = param_.tokenOffset.size();
  if (batch_size == 1) {
    BuildGraphWithMulsBS1();
  } else {
    BuildGraphWithMuls();
  }

  SetKernelGrapModifyFunc();
}

SelfAttentionKvCacheFusionOpsGPT3Runner::~SelfAttentionKvCacheFusionOpsGPT3Runner() {}

void SelfAttentionKvCacheFusionOpsGPT3Runner::BuildGraphWithMuls()
{
  size_t tensorId = 0;

  // kv cache input
  kernelGraph_.inTensors.resize(7);
  AsdOps::Tensor &mixedLayer = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
  // flash attention input
  AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);

  kernelGraph_.outTensors.resize(1);
  AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);

  size_t internalTensorId = 0;
  kernelGraph_.internalTensors.resize(6);
  AsdOps::Tensor &transposedMix = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedQuery = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedKey = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedValue = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &attOut = kernelGraph_.internalTensors.at(internalTensorId++);

  size_t nodeId = 0;
  kernelGraph_.nodes.resize(7);
  auto &permuteMixNode = kernelGraph_.nodes.at(nodeId++);
  auto &splitNode = kernelGraph_.nodes.at(nodeId++);
  auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
  auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
  auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
  auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
  auto &permuteflashOutNode = kernelGraph_.nodes.at(nodeId++);

  AsdOps::OpParam::Transpose permuteMixNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
  permuteMixNode.opDesc = {0, "TransposeOperation", permuteMixNodeParam};
  permuteMixNode.inTensors = {&mixedLayer};
  permuteMixNode.outTensors = {&transposedMix};

  /* [seq_len, bs, 3 * hidden_size] -> [seq_len, bs, num_head, 3 * head_dim] ->split  */
  splitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{3, 3}};
  splitNode.inTensors = {&transposedMix};
  splitNode.outTensors = {&mixedQuery, &mixedKey, &mixedValue};
  splitNode.inTensorViewFuncs.resize(splitNode.inTensors.size());
  splitNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, 3 * param_.dk};
  };
  splitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
    runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 1、k cache
  kCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
  kCacheNode.inTensors = {&mixedKey, &layerId, &cacheK, &tokenOffset, &seqLen};
  kCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space
  kCacheNode.inTensorViewFuncs.resize(kCacheNode.inTensors.size());
  kCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };
  kCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 2、V cache  seq_len, batch, head_num, head_size]
  vCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
  vCacheNode.inTensors = {&mixedValue, &layerId, &cacheV, &tokenOffset, &seqLen};
  vCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space
  vCacheNode.inTensorViewFuncs.resize(vCacheNode.inTensors.size());
  vCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };
  vCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 3、div
  float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
  mulsQNode.opDesc = {0, "ElewiseOperation",
                      AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
  mulsQNode.inTensors = {&mixedQuery};
  mulsQNode.outTensors = {&divOut};
  mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /*
   * 4、flash attention
   * input:
   * q:[ntokens, hidden_size]
   * k_present: [layer, batch, max_seqlen, hidden_size]
   * v_present:[layer, batch, max_seqlen, hidden_size]
   * layer_id: [1]
   * attenion_mask: [max_seqlen, max_seqlen]
   * output:[ntokens, hidden_size]
   */
  float tor = (float)(param_.layerId + 1);
  flashAttentionNode.opDesc = {0, "AttentionOperation",
                                AsdOps::OpParam::Attention{param_.headNum, param_.seqLen, param_.tokenOffset, tor}};

  flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, &attentionMask};
  flashAttentionNode.outTensors = {&attOut};
  flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
  flashAttentionNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };
  /* [ntokens, hidden_size]-> [seq_len, batch_size, hidden_size] -> [batch_size, seq_len, hidden_size] */
  AsdOps::OpParam::Transpose permuteflashOutNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
  permuteflashOutNode.opDesc = {0, "TransposeOperation", permuteflashOutNodeParam};
  permuteflashOutNode.inTensors = {&attOut};
  permuteflashOutNode.outTensors = {&context};
  permuteflashOutNode.inTensorViewFuncs.resize(permuteflashOutNode.inTensors.size());
  permuteflashOutNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
    newDims = {1, oldDims.at(0), oldDims.at(1)}; /* 增量seq_len固定为1 */
  };
}

void SelfAttentionKvCacheFusionOpsGPT3Runner::BuildGraphWithMulsBS1()
{
  size_t tensorId = 0;
  // kv cache input
  kernelGraph_.inTensors.resize(7);
  AsdOps::Tensor &mixedLayer = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
  // flash attention input
  AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
  AsdOps::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);

  kernelGraph_.outTensors.resize(1);
  AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);


  size_t internalTensorId = 0;
  kernelGraph_.internalTensors.resize(5);
  AsdOps::Tensor &transposedMix = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedQuery = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedKey = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &mixedValue = kernelGraph_.internalTensors.at(internalTensorId++);
  AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorId++);

  size_t nodeId = 0;
  kernelGraph_.nodes.resize(6);
  auto &permuteMixNode = kernelGraph_.nodes.at(nodeId++);
  auto &splitNode = kernelGraph_.nodes.at(nodeId++);
  auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
  auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
  auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
  auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);

  AsdOps::OpParam::Transpose permuteMixNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
  permuteMixNode.opDesc = {0, "TransposeOperation", permuteMixNodeParam};
  permuteMixNode.inTensors = {&mixedLayer};
  permuteMixNode.outTensors = {&transposedMix};

  /* [seq_len, bs, 3 * hidden_size] -> [seq_len, bs, num_head, 3 * head_dim] ->split  */
  splitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{3, 3}};
  splitNode.inTensors = {&transposedMix};
  splitNode.outTensors = {&mixedQuery, &mixedKey, &mixedValue};
  splitNode.inTensorViewFuncs.resize(splitNode.inTensors.size());
  splitNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, 3 * param_.dk};
  };
  splitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
    runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 1、k cache
  kCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
  kCacheNode.inTensors = {&mixedKey, &layerId, &cacheK, &tokenOffset, &seqLen};
  kCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space
  kCacheNode.inTensorViewFuncs.resize(kCacheNode.inTensors.size());
  kCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };
  kCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 2、V cache  seq_len, batch, head_num, head_size]
  vCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
  vCacheNode.inTensors = {&mixedValue, &layerId, &cacheV, &tokenOffset, &seqLen};
  vCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space
  vCacheNode.inTensorViewFuncs.resize(vCacheNode.inTensors.size());
  vCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };
  vCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 3、div
  float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
  mulsQNode.opDesc = {0, "ElewiseOperation",
                      AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
  mulsQNode.inTensors = {&mixedQuery};
  mulsQNode.outTensors = {&divOut};
  mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo)
  {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++)
    {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  // 4、flash attention
  float tor = (float)(param_.layerId + 1);
  flashAttentionNode.opDesc = {0, "AttentionOperation",
                                AsdOps::OpParam::Attention{param_.headNum, param_.seqLen, param_.tokenOffset, tor}};

  flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, &attentionMask};
  flashAttentionNode.outTensors = {&context};
  flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
  flashAttentionNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims)
  {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
  };

}

void SelfAttentionKvCacheFusionOpsGPT3Runner::SetKernelGrapModifyFunc()
{
  kernelGraph_.kernelGraphModifyFunc = [&](const RunnerVariantPack &runnerVariantPack)
  {
    if (typeid(SelfAttentionKvCacheFusionVariantPackParam) != runnerVariantPack.param.Type())
    {
      ASD_LOG(FATAL) << "SelfAttentionKvCacheFusionOpsGPT3Runner invalid type "
                        "SelfAttentionKvCacheFusionVariantPackParam, Current:"
                      << runnerVariantPack.param.Type().name() << " type id :" << typeid(SelfAttentionKvCacheFusionVariantPackParam).name();
      return;
    }
    const SelfAttentionKvCacheFusionVariantPackParam &newParam =
        AsdOps::AnyCast<SelfAttentionKvCacheFusionVariantPackParam>(runnerVariantPack.param);
    const size_t flashAttentionNodeId = 5; // 注意为flashAttention node 的index，如果有改动需要，一定要适配！
    auto &flashAttentionNode = kernelGraph_.nodes.at(flashAttentionNodeId);

    flashAttentionNode.opDesc = {
        0, "AttentionOperation",
        AsdOps::OpParam::Attention{param_.headNum, newParam.seqLen, newParam.tokenOffset, (float)(param_.layerId + 1)}};

    ASD_LOG(INFO) << "SelfAttentionKvCacheFusionOpsGPT3Runner SetOpDesc AsdOps::OpParam::Attention.headNum:"
                  << param_.headNum << ", seqLen:" << newParam.seqLen << ", tokenOffset:" << newParam.tokenOffset;
  };
}
} // namespace AclTransformer
