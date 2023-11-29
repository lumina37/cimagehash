#pragma once

#include "macros.h"
#include "consts.hpp"

namespace igh::inline structs {

template <typename T> class BGR
{
public:
    BGR() = default;
    BGR(const BGR& rhs) = default;

    BGR operator+(const BGR& rhs) { return {b + rhs.b, g + rhs.g, r + rhs.r}; }

    T IGH_FORCEINLINE gray() const { return b * ABLUE + g * AGREEN + r * ARED; }

    static constexpr int lanes = 3;

    T b;
    T g;
    T r;
};

using BGRu32 = BGR<uint32_t>;

} // namespace igh::inline structs