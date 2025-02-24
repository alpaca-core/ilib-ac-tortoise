// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Lib.hpp>
#include <ac/local/IoCtx.hpp>
#include <ac/schema/BlockingIoHelper.hpp>
#include <ac/schema/FrameHelpers.hpp>

#include <ac/schema/TortoiseCpp.hpp>

#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/DefaultSink.hpp>

#include <iostream>

#include "ac-test-data-tortoise-dir.h"
#include "aclp-tortoise-info.h"

#include <ac-audio.hpp>

namespace schema = ac::schema::tortoise;

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::DefaultSink>();

    ac::local::Lib::loadPlugin(ACLP_tortoise_PLUGIN_FILE);

    ac::frameio::BlockingIoCtx blockingCtx;
    ac::local::IoCtx io;

    auto& provider = ac::local::Lib::getProvider("tortoise.cpp");
    ac::schema::BlockingIoHelper tortoise(io.connect(provider), blockingCtx);

    tortoise.expectState<schema::StateInitial>();
    tortoise.call<schema::StateInitial::OpLoadModel>({
        .aggresivePath = AC_TEST_DATA_TORTOISE_DIR "/ggml-model.bin",
        .diffusionPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-diffusion-model.bin",
        .vocoderPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-vocoder-model.bin"
    });
    tortoise.expectState<schema::StateModelLoaded>();
    tortoise.call<schema::StateModelLoaded::OpStartInstance>({
        .tokenizerPath = AC_TEST_DATA_TORTOISE_DIR "/tokenizer.json",
    });

    tortoise.expectState<schema::StateInstance>();
    auto result = tortoise.call<schema::StateInstance::OpTTS>({
        .text = "This is an example of Tortoise Alpaca plugin.",
        .voicePath = AC_TEST_DATA_TORTOISE_DIR "/mouse.bin",
    });

    // auto res = ac::Dict_optValueAt(result, "result", ac::Blob{});
    float* resData = reinterpret_cast<float*>(result.audioData.value().data());
    size_t resSize = result.audioData.value().size() / sizeof(float);

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
