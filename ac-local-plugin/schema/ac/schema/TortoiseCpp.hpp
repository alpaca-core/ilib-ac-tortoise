// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include <ac/schema/Field.hpp>
#include <ac/Dict.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace ac::schema {

inline namespace tortoise {

struct StateInitial {
    static constexpr auto id = "initial";
    static constexpr auto desc = "Initial state";

    struct OpLoadModel {
        static constexpr auto id = "load-model";
        static constexpr auto desc = "Load the tortoise.cpp model";

        struct Params{
            Field<std::string> aggresivePath = std::nullopt;
            Field<std::string> diffusionPath = std::nullopt;
            Field<std::string> vocoderPath = std::nullopt;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(aggresivePath, "aggresivePath", "Path to the file with aggresive model.");
                v(diffusionPath, "diffusionPath", "Path to the file with diffusion model.");
                v(vocoderPath, "vocoderPath", "Path to the file with vocoder model.");
            }
        };

        using Return = nullptr_t;
    };

    using Ops = std::tuple<OpLoadModel>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateModelLoaded {
    static constexpr auto id = "model-loaded";
    static constexpr auto desc = "Model loaded state";

    struct OpStartInstance {
        static constexpr auto id = "start-instance";
        static constexpr auto desc = "Start a new instance of the whisper.cpp model";

        struct Params{
            Field<std::string> tokenizerPath = std::nullopt;
            Field<unsigned> seed = Default(42);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(tokenizerPath, "tokenizerPath", "Path to the file with tokenizer.");
                v(seed, "seed", "Seed for the model");
            }
        };

        using Return = nullptr_t;
    };

    using Ops = std::tuple<OpStartInstance>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateInstance {
    static constexpr auto id = "instance";
    static constexpr auto desc = "Instance state";

    struct OpTTS {
        static inline constexpr std::string_view id = "tts";
        static inline constexpr std::string_view desc = "Run the tortoise.cpp inference and produce audio";

        struct Params {
            Field<std::string> text;
            Field<std::string> voicePath;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(text, "text", "Text to generate audio from");
                v(voicePath, "voicePath", "Path to the voice model");
            }
        };

        struct Return {
            Field<ac::Blob> audioData;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(audioData, "audioData", "Audio of the text");
            }
        };
    };

    using Ops = std::tuple<OpTTS>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct Interface {
    static inline constexpr std::string_view id = "tortoise.cpp";
    static inline constexpr std::string_view desc = "ilib-ac-tortoise-specific interface";

    using States = std::tuple<StateInitial, StateModelLoaded, StateInstance>;
};

} // namespace tortoise

} // namespace ac::schema
