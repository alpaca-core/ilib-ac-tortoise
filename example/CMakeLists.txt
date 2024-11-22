# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
set(TARGET example-ac-tortoise-basic)

add_executable(${TARGET} e-basic.cpp)
target_link_libraries(${TARGET} PRIVATE
    ac::tortoise
    ac::jalog
    ac-test-data::tortoise
    ac-dev::audio
)
set_target_properties(${TARGET} PROPERTIES FOLDER example)
