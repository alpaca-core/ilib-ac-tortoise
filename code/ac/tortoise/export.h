// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include <splat/symbol_export.h>

#if AC_TORTOISE_SHARED
#   if BUILDING_AC_TORTOISE
#       define AC_TORTOISE_EXPORT SYMBOL_EXPORT
#   else
#       define AC_TORTOISE_EXPORT SYMBOL_IMPORT
#   endif
#else
#   define AC_TORTOISE_EXPORT
#endif
