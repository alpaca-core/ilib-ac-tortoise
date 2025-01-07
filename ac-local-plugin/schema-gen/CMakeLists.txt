# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_executable(ac-tortoise.cpp-schema-gen)
target_sources(ac-tortoise.cpp-schema-gen PRIVATE
    schema-gen.cpp
)
target_link_libraries(ac-tortoise.cpp-schema-gen PRIVATE
    ac::tortoise.cpp-schema
)
