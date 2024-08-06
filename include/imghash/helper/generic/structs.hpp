#pragma once

#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/macros.h"

namespace igh::inline structs {

class BlockSumU32
{
public:
    BlockSumU32() = default;
    BlockSumU32(const BlockSumU32& rhs) = default;

    [[nodiscard]] IGH_FORCEINLINE uint32_t gray() const { return b_ * ABLUE + g_ * AGREEN + r_ * ARED; }

    static constexpr uint CHANNELS = 3;

    uint32_t b_, g_, r_;
};

} // namespace igh::inline structs