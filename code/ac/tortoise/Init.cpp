// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Init.hpp"
#include "Logging.hpp"
#include <ac/tortoise/tortoise.hpp>

namespace ac::tortoise {

namespace {
static void tortoiseLogCb(ggml_log_level level, const char* text, void* /*user_data*/) {
    auto len = strlen(text);

    auto jlvl = [&]() {
        switch (level) {
        case GGML_LOG_LEVEL_ERROR: return jalog::Level::Error;
        case GGML_LOG_LEVEL_WARN: return jalog::Level::Warning;
        case GGML_LOG_LEVEL_INFO: return jalog::Level::Info;
        case GGML_LOG_LEVEL_DEBUG: return jalog::Level::Debug;
        default: return jalog::Level::Critical;
        }
    }();

    // skip newlines from llama, as jalog doen't need them
    if (len > 0 && text[len - 1] == '\n') {
        --len;
    }

    log::scope.addEntry(jlvl, {text, len});
}
}


void initLibrary() {
    // tortoise_log_set(tortoiseLogCb, nullptr);
}

} // namespace ac::tortoise
