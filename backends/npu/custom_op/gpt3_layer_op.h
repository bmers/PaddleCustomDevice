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

struct ChatGlm6BLayerParam {
    double layerNormEps = 0;
    int headNum = 0;
    bool transKey = false;
    int dk = 0;
    int layerId = 0;
    float residualAddScale = 0;
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