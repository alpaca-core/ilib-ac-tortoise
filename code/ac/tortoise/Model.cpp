// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Model.hpp"
#include <ac/tortoise/tortoise.hpp>

#include <astl/move.hpp>
#include <stdexcept>

namespace ac::tortoise {
namespace {

}

Model::Model(std::string_view autoregressiveModelPath,
    std::string_view diffusionModelPath,
    std::string_view vocoderModelPath,
    Params params)
    : m_params(astl::move(params))
    , m_autoregressiveModel(autoregressive_model_load(autoregressiveModelPath.data()), free_autoregressive_model)
    , m_diffusionModel(diffusion_model_load(diffusionModelPath.data()), free_diffusion_model)
    , m_vocoderModel(vocoder_model_load(vocoderModelPath.data()), free_vocoder_model)
{}


} // namespace ac::tortoise
