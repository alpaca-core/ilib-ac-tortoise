// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// trivial example of using alpaca-core's tortoise inference

// tortoise
#include <ac/tortoise/Init.hpp>
#include <ac/tortoise/Model.hpp>
#include <ac/tortoise/Instance.hpp>

#include <ac-audio.hpp>

// logging
#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-tortoise-dir.h"

#include <iostream>
#include <string>
#include <vector>

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    std::cout << "Basic example of tortoise\n";

    // initialize the library
    ac::tortoise::initLibrary();

    // load model
    std::string autoregressiveModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-model.bin";
    std::string diffusionModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-diffusion-model.bin";
    std::string vocoderModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-vocoder-model.bin";
    ac::tortoise::Model model(autoregressiveModelPath, diffusionModelPath, vocoderModelPath, {});

    std::string tokenizerPath = AC_TEST_DATA_TORTOISE_DIR "/tokenizer.json";

    // // create inference instance
    ac::tortoise::Instance instance(model, { .tokenizerPath = tokenizerPath });

    std::string voicePath = AC_TEST_DATA_TORTOISE_DIR "/mouse.bin";

    // transcript the audio
    auto res = instance.textToSpeech("Hello my friend, how are you today?", voicePath);

    ac::audio::WavWriter wavWriter;
    wavWriter.open("output.wav", 24000, 16, 1);
    wavWriter.write(res.data(), res.size());
    wavWriter.close();

    std::cout << "Output written to output.wav\n";

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
