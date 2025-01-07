# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_ac_local_plugin(
    NAME tortoise
    SOURCES
        LocalTortoise.cpp
    LIBRARIES
        ac::tortoise
        ac::tortoise.cpp-schema
)
