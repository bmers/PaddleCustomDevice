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

#include "atb/atb_infer.h"
#include "atb_layer_base.h"
#include "paddle/phi/extension.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"
#include "paddle/phi/backends/c_comm_lib.h"
namespace phi {
namespace detail {
ccl::CCLComm GetCCLComm(const Place& place, int global_gid);
}
}

#define ATB_FLASH_ATTENTION_MAX_SEQ_LEN 1024

class PpAtbLlaMaDecoderLayerParallelOp : public PpAscendAtbLayerOpBase {
public:
  PpAtbLlaMaDecoderLayerParallelOp(const std::string &modelName, int32_t layerNum);
  ~PpAtbLlaMaDecoderLayerParallelOp();

private:
  void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                        std::vector<const phi::DenseTensor *> &outTensors);

};

class PpAtbLlaMaEncoderLayerParallelOp : public PpAscendAtbLayerOpBase {
public:
  PpAtbLlaMaEncoderLayerParallelOp(const std::string &modelName, int32_t layerNum);
  ~PpAtbLlaMaEncoderLayerParallelOp();
};
