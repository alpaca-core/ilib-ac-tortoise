// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Instance.hpp"
#include "Model.hpp"
#include "Logging.hpp"

#include <ac/tortoise/tortoise.hpp>

#include <astl/throw_stdex.hpp>
#include <astl/iile.h>
#include <astl/move.hpp>
#include <itlib/sentry.hpp>

#include <regex>

namespace ac::tortoise {

Instance::Instance(Model& model, InitParams params)
    : m_model(model)
    , m_params(astl::move(params))
    , m_vocab(new gpt_vocab)
{
    gpt_vocab_init(m_params.tokenizerPath.c_str(), *m_vocab);
}

std::vector<float> Instance::textToSpeech(std::string_view text, std::string_view voicePath) {
    std::string textStr = std::regex_replace(std::string(text), std::regex(" "), "[SPACE]");

    static std::vector<gpt_vocab::id> pre = ::parse_tokens_from_string("255", ',');
    static std::vector<gpt_vocab::id> post = ::parse_tokens_from_string("0", ',');

    std::vector<gpt_vocab::id> tokens = ::gpt_tokenize(*m_vocab, textStr);

    tokens.insert(tokens.end(), post.begin(), post.end());
    tokens.insert(tokens.begin(), pre.begin(), pre.end());

    auto autoregressiveResult = autoregressive(*m_model.autoregressiveModel(), tokens, voicePath.data(), 1);

    trimmed_latents_vector& trimmedLatents = autoregressiveResult.first;
    // sequence_vector& sequences = autoregressiveResult.second;

    std::vector<float> mel = diffusion(*m_model.diffusionModel(), trimmedLatents[0]);
    std::vector<float> audio = vocoder(*m_model.vocoderModel(), mel);

    return audio;
}

} // namespace ac::tortoise
