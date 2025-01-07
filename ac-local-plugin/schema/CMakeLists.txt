# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_library(ac-tortoise.cpp-schema INTERFACE)
add_library(ac::tortoise.cpp-schema ALIAS ac-tortoise.cpp-schema)
target_link_libraries(ac-tortoise.cpp-schema INTERFACE ac::schema)
target_include_directories(ac-tortoise.cpp-schema INTERFACE .)
