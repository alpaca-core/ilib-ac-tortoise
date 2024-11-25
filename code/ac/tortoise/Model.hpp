// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <astl/mem_ext.hpp>

#include <string>

struct autoregressive_model;
struct diffusion_model;
struct vocoder_model;
namespace ac::tortoise {

class AC_TORTOISE_EXPORT Model {
public:
    struct Params {
    };

    Model(std::string_view autoregressiveModelPath,
        std::string_view diffusionModelPath,
        std::string_view vocoderModelPath,
        Params params);
    ~Model() = default;

    const Params& params() const noexcept { return m_params; }

    autoregressive_model* autoregressiveModel() const noexcept { return m_autoregressiveModel.get(); }
    diffusion_model* diffusionModel() const noexcept { return m_diffusionModel.get(); }
    vocoder_model* vocoderModel() const noexcept { return m_vocoderModel.get(); }

private:
    const Params m_params;
    astl::c_unique_ptr<autoregressive_model> m_autoregressiveModel;
    astl::c_unique_ptr<diffusion_model> m_diffusionModel;
    astl::c_unique_ptr<vocoder_model> m_vocoderModel;
};
} // namespace ac::tortoise
