# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
function(add_tortoise_plugin_example name)
    set(tgt example-aclp-tortoise-${name})
    add_executable(${tgt} ep-${name}.cpp)
    target_link_libraries(${tgt} PRIVATE
        ac::local
        ac::jalog
        ac-test-data::tortoise
        aclp::tortoise-info
        ac-dev::audio
        ac::tortoise.cpp-schema
    )
    set_target_properties(${tgt} PROPERTIES FOLDER example)
endfunction()

add_tortoise_plugin_example(tts)


