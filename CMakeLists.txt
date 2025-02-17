# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(ilib-tortoise.cpp
    VERSION 1.0.0
    DESCRIPTION "Example plugin for the Alpaca Core Local SDK"
    LANGUAGES CXX
)

include(./get-ac-build.cmake)

#################
# cmake lib
CPMAddPackage(gh:iboB/icm@1.5.4)
list(APPEND CMAKE_MODULE_PATH
    "${icm_SOURCE_DIR}"
)

include(icm_add_lib)

include(init_ac_prj)

include(ac_build_prj_util)
include(ac_build_plugin_util)

#################
# options

option(AC_TORTOISE_BUILD_TESTS "${PROJECT_NAME}: build tests" ${testsDefault})
option(AC_TORTOISE_BUILD_EXAMPLES "${PROJECT_NAME}: build examples" ${examplesDefault})
mark_as_advanced(AC_TORTOISE_BUILD_TESTS AC_TORTOISE_BUILD_EXAMPLES)

init_ac_plugin_option(TORTOISE)

#######################################
# packages

add_ac_local(0.1.12)
CPMAddPackage(gh:iboB/splat@1.3.3)
CPMAddPackage(gh:iboB/itlib@1.11.4)

#######################################
# submodule and config

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(haveCuda YES)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "${CMAKE_PROJECT_NAME}: system CUDA found")
    # we should add -forward-unknown-to-host-compiler but it somehow works without it
    # could it be that it depends on the CMake version?
endif()

if(haveCuda)
    set(GGML_CUBLAS ON)
endif()

if(UNIX AND APPLE)
    set(GGML_METAL ON)
endif()

set(GGML_CCACHE OFF)

add_subdirectory(ggml)

#######################################
# subdirs
add_subdirectory(code)

if(AC_TORTOISE_BUILD_TESTS OR AC_TORTOISE_BUILD_EXAMPLES)
    CPMAddPackage(
        NAME ac-test-data-tortoise
        VERSION 1.0.0
        GIT_REPOSITORY https://huggingface.co/alpaca-core/ac-test-data-tortoise
        GIT_TAG feec6ba7f383286eaefafb22ce25c7e89c7c1438
    )
    CPMAddPackage(gh:alpaca-core/helper-audio@1.0.0)
endif()

if(AC_TORTOISE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(AC_TORTOISE_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_AC_TORTOISE_PLUGIN)
    add_subdirectory(ac-local-plugin)
endif()
