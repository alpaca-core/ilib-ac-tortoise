// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <astl/mem_ext.hpp>

#include <string>

namespace ac::tortoise {
class Job;

class AC_TORTOISE_EXPORT Model {
public:
    struct Params {
        bool gpu = true; // try to load data on gpu
    };

    Model(const char* pathToBin, Params params);
    ~Model();

    const Params& params() const noexcept { return m_params; }

    // whisper_context* context() const noexcept { return m_ctx.get(); }

private:
    const Params m_params;
    // astl::c_unique_ptr<whisper_context> m_ctx;
};
} // namespace ac::tortoise
