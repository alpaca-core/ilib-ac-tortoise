// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Model.hpp"
#include <ac/tortoise/tortoise.hpp>
#include <astl/move.hpp>
#include <stdexcept>

namespace ac::tortoise {

Model::Model(const char* pathToBin, Params params)
    : m_params(astl::move(params))
    // , m_ctx(whisper_init_from_file_with_params_no_state(pathToBin, whisperFromModelParams(m_params)), whisper_free)
{
    // if (!m_ctx) {
    //     throw std::runtime_error("Failed to load model");
    // }
}

Model::~Model() = default;


} // namespace ac::tortoise
