// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <astl/mem_ext.hpp>

#include <functional>
#include <string>
#include <span>

#include <ac/tortoise/common.h>


namespace ac::tortoise {
class Model;

class AC_TORTOISE_EXPORT Instance {
public:
    struct InitParams {
        std::string tokenizerPath;
    };

    Instance(Model& model, InitParams params);
    ~Instance() = default;

    std::vector<float> textToSpeech(std::string_view text, std::string_view voicePath);
private:

    Model& m_model;
    InitParams m_params;
    std::unique_ptr<gpt_vocab> m_vocab;
};

} // namespace ac::tortoise
