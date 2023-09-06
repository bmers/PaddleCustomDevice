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
#include <acl/acl.h>
#include "atb/atb_infer.h"
#include "paddle/extension.h"

class PpAscendAtbLayerOpBase {
public:
  PpAscendAtbLayerOpBase(const std::string &opName, int32_t layerNum);
  ~PpAscendAtbLayerOpBase();

  virtual void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                std::vector<const phi::DenseTensor *> &outTensors);
  atb::Status Execute(aclrtStream stream,
                      std::vector<const phi::DenseTensor *> &inTensors,
                      std::vector<const phi::DenseTensor *> &outTensors);

  std::shared_ptr<atb::Operation> operation_;

protected:
  std::string opName_;
  atb::VariantPack variantPacks_;

private:
  void SetWorkspace(uint64_t workspace_size);

private:
  uint64_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  int32_t curBatchSize_ = 0;
  int32_t layerNum_ = 0;

  int32_t layerCount_;

  uint64_t executeCount_ = 0;
  int32_t currentDevId_ = 0;

  aclrtStream stream_;
  std::vector<phi::DenseTensor> layerIdTensor;
};
