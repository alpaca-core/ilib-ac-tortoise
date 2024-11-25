# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
CPMAddPackage(gh:iboB/doctest-util@0.1.3)
set_target_properties(doctest PROPERTIES FOLDER test)
set_target_properties(doctest-main PROPERTIES FOLDER test)

add_doctest_lib_test(integration ac-tortoise
    SOURCES
        t-integration.cpp
    LIBRARIES
        ac-test-data::tortoise
)
