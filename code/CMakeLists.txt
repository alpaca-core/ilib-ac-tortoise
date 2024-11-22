# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_library(tortoise STATIC
    ac/tortoise/common.h
    ac/tortoise/common.cpp
    ac/tortoise/tortoise.hpp
    ac/tortoise/tortoise.cpp
)

target_include_directories(tortoise INTERFACE .)

target_link_libraries(tortoise
    PUBLIC
        ggml
)

icm_add_lib(ac-tortoise AC_WHISPER)
add_library(ac::tortoise ALIAS ac-tortoise)
target_include_directories(ac-tortoise INTERFACE .)
target_link_libraries(ac-tortoise
    PRIVATE
        tortoise
        ac::jalog
    PUBLIC
        splat::splat
        ac::astl
        itlib::itlib
)
target_sources(ac-tortoise PRIVATE
    ac/tortoise/export.h
    ac/tortoise/Init.hpp
    ac/tortoise/Init.cpp
    ac/tortoise/Logging.hpp
    ac/tortoise/Logging.cpp
    ac/tortoise/Model.hpp
    ac/tortoise/Model.cpp
    ac/tortoise/Instance.hpp
    ac/tortoise/Instance.cpp
)
