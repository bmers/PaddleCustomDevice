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

#pragma once

// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "acltransformer/graph_operation.h"
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#define GPT3_LAYER_FLASH_ATTENTION_MAX_SEQ_LEN 1024

namespace AclTransformer {
struct GPT3LayerParam {
    float layerNormEps = 0;
    int layerNormBeginNormAxis = 2;
    int head_dim = 0;
    int head_num = 0;
    int layer_num = 0;
    AsdOps::SVector<int32_t> seqLen;
    AsdOps::SVector<int32_t> tokenOffset;
    HcclComm comm;
};

class GPT3LayerDecoderOperation : public GraphOperation {
 public:
  explicit GPT3LayerDecoderOperation(const GPT3LayerParam &param);
  ~GPT3LayerDecoderOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  GPT3LayerParam param_;
};

class GPT3LayerWithoutCacheDecoderOperation : public GraphOperation {
 public:
  explicit GPT3LayerWithoutCacheDecoderOperation(const GPT3LayerParam &param);
  ~GPT3LayerWithoutCacheDecoderOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  GPT3LayerParam param_;
};

class GPT3LayerDecoderParallelOperation : public GraphOperation {
 public:
  explicit GPT3LayerDecoderParallelOperation(const GPT3LayerParam &param);
  ~GPT3LayerDecoderParallelOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  GPT3LayerParam param_;
};

class GPT3LayerWithoutCacheDecoderParallelOperation : public GraphOperation {
 public:
  explicit GPT3LayerWithoutCacheDecoderParallelOperation(const GPT3LayerParam &param);
  ~GPT3LayerWithoutCacheDecoderParallelOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  GPT3LayerParam param_;
};
} // namespace AclTransformer

struct GPT3LayerWorkspace {
    void *workspace_ = nullptr;
    uint64_t workspaceSize_ = 0;
};

class GPT3LayerParallelCustomOp {
public:
  GPT3LayerParallelCustomOp(int layerNum, int batchSize, AclTransformer::Handle handle);
  void SetParam(AclTransformer::GPT3LayerParam &param);
  AclTransformer::VariantPack &GetVariantPack(int layerId);
  void Setup(int layerId, AclTransformer::Handle handle);
  void SyncFinalLayer(aclrtStream stream);
  int GetCurBatchSize();

private:
  void ThreadProcessTask();
  void PushTask(int layerId);
  int PopTask();
  void ExecutePlan(int layerId);

private:
  int currentDevId_ = 0;
  int curBatchSize_ = 0;
  int layerNum_ = 0;
  std::vector<AclTransformer::VariantPack> variantPacks_;
  std::vector<std::shared_ptr<AclTransformer::GraphOperation>> operations_;
  std::vector<std::shared_ptr<AclTransformer::Plan>> plans_;
  std::thread taskProcessThread_;
  std::queue<int> taskQueue_;
  std::atomic_bool allTaskFinish_;
  std::atomic_int layerCount_;
  std::mutex mutex_;
  std::condition_variable cond_;
  AclTransformer::Handle handle_;
};
#endif // PADDLE_WITH_ASCEND_TRANSFORMER_ACC
