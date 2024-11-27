// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Model.hpp>
#include <ac/local/Instance.hpp>
#include <ac/local/ModelLoaderRegistry.hpp>
#include <ac/local/Lib.hpp>

#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/DefaultSink.hpp>

#include <iostream>

#include "ac-test-data-tortoise-dir.h"
#include "aclp-tortoise-info.h"

#include <ac-audio.hpp>

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::DefaultSink>();

    ac::local::Lib::loadPlugin(ACLP_tortoise_PLUGIN_FILE);;

    // load model
    auto model = ac::local::Lib::loadModel(
        {
            .type = "tortoise.cpp",
            .assets = {
                {.path = AC_TEST_DATA_TORTOISE_DIR "/ggml-model.bin"},
                {.path = AC_TEST_DATA_TORTOISE_DIR "/ggml-diffusion-model.bin"},
                {.path = AC_TEST_DATA_TORTOISE_DIR "/ggml-vocoder-model.bin"}
            }
        },
        {}, // no params
        {} // empty progress callback
    );
    ac::Dict params;
    params["tokenizerPath"] = AC_TEST_DATA_TORTOISE_DIR "/tokenizer.json";
    params["seed"] = 42;
    auto instance = model->createInstance("general", params);

    auto result = instance->runOp("tts", {
            {"text", "This is an example of Tortoise Alpaca plugin."},
            {"voicePath", AC_TEST_DATA_TORTOISE_DIR "/mouse.bin"},
        }, {});

    auto res = ac::Dict_optValueAt(result, "result", ac::Blob{});
    float* resData = reinterpret_cast<float*>(res.data());
    size_t resSize = res.size() / sizeof(float);

    ac::audio::WavWriter wavWriter;
    wavWriter.open("pluginOutput.wav", 24000, 16, 1);
    wavWriter.write(resData, resSize);
    wavWriter.close();

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
