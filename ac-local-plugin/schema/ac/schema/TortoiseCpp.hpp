// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include <ac/schema/Field.hpp>
#include <ac/Dict.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace ac::local::schema {

struct TortoiseCppInterface {
    static inline constexpr std::string_view id = "tortoise.cpp";
    static inline constexpr std::string_view description = "ilib-ac-tortoise.cpp-specific interface";

    struct OpTTS {
        static inline constexpr std::string_view id = "tts";
        static inline constexpr std::string_view description = "Run the tortoise.cpp inference and produce audio";

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
            Field<ac::Blob> result;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(result, "result", "Audio of the text");
            }
        };
    };

    using Ops = std::tuple<OpTTS>;
};

struct TortoiseCppLoader {
    static inline constexpr std::string_view id = "tortoise.cpp";
    static inline constexpr std::string_view description = "Inference based on our fork of https://github.com/ggerganov/tortoise.cpp";

    using Params = nullptr_t;

    struct InstanceGeneral {
        static inline constexpr std::string_view id = "general";
        static inline constexpr std::string_view description = "General instance";

        using Params = nullptr_t;

        using Interfaces = std::tuple<TortoiseCppInterface>;
    };

    using Instances = std::tuple<InstanceGeneral>;
};

} // namespace ac::local::schema
