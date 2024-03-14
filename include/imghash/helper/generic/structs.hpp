#pragma once

#include "imghash/helper/common/consts.hpp"

namespace igh::inline structs {

template <typename T> class BlockSum
{
public:
    BlockSum() = default;
    BlockSum(const BlockSum& rhs) = default;

    [[nodiscard]] IGH_FORCEINLINE T gray() const { return b_ * ABLUE + g_ * AGREEN + r_ * ARED; }

    static constexpr uint CHANNELS = 3;

    T b_, g_, r_;
};

using BlockSumU32 = BlockSum<uint32_t>;

} // namespace igh::inline structs