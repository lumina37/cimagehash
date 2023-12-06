#pragma once

#include <cstdint>

#include "types.hpp"
#include "cfunc.hpp"

namespace igh::inline consts {

constexpr uint32_t AMP_RS = 7; // right shift factor of global AMP
constexpr uint32_t AMP = (uint32_t)1 << AMP_RS;

// follows BT.601
constexpr double RED = 0.299;
constexpr double GREEN = 0.587;
constexpr double BLUE = 0.114;
constexpr uint32_t ARED = static_cast<uint32_t>(cround(RED * AMP));
constexpr uint32_t AGREEN = static_cast<uint32_t>(cround(GREEN * AMP));
constexpr uint32_t ABLUE = static_cast<uint32_t>(cround(BLUE * AMP));

constexpr uint DST_W = 8;
constexpr uint DST_H = 8;
constexpr uint HASH_LEN = DST_H * DST_W;

} // namespace igh::inline consts