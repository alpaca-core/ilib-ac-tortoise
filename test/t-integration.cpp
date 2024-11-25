// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/tortoise/Init.hpp>
#include <ac/tortoise/Model.hpp>
#include <ac/tortoise/Instance.hpp>

#include <doctest/doctest.h>

#include "ac-test-data-tortoise-dir.h"

#include <iostream>
#include <fstream>

struct GlobalFixture {
    GlobalFixture() {
        ac::tortoise::initLibrary();
    }
};

GlobalFixture globalFixture;

const char* autoregressiveModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-model.bin";
const char* diffusionModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-diffusion-model.bin";
const char* vocoderModelPath = AC_TEST_DATA_TORTOISE_DIR "/ggml-vocoder-model.bin";
const char* tokenizerPath = AC_TEST_DATA_TORTOISE_DIR "/tokenizer.json";
const char* voicePath = AC_TEST_DATA_TORTOISE_DIR "/mouse.bin";

void generateHeaderWithData(const char* path, const char* variableName, const std::vector<float>& data) {
    std::ofstream f(path);
    f << "#pragma once\n";
    f << "#include <vector>\n";
    f << "std::vector<float> "<< variableName <<" = {\n\t";

    for (size_t i = 0; i < data.size(); i++) {
        f << data[i];
        f << (i < data.size() - 1 ? "," : "");
    }
    f << "\n};\n";

    f.close();
}

TEST_CASE("inference") {
    ac::tortoise::Model model(autoregressiveModelPath, diffusionModelPath, vocoderModelPath, {});
    REQUIRE(!!model.autoregressiveModel());
    REQUIRE(!!model.diffusionModel());
    REQUIRE(!!model.vocoderModel());

    // general inference
    {
        ac::tortoise::Instance instance(model, { .tokenizerPath = tokenizerPath });

        auto res = instance.textToSpeech("This is alpaca test.", voicePath);
        CHECK(res.size() > 0);

        // If there is a change in the model, uncomment the following line to update the expected result
        // generateHeaderWithData("InferenceResult.h", "g_testResult", res);

        #include "InferenceResult.h"

        CHECK(res.size() == g_testResult.size());

        for (size_t i = 0; i < res.size(); i++)
        {
            CHECK(abs(res[i] - g_testResult[i]) < .000001);
        }
    }
}
