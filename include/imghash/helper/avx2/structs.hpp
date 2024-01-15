#pragma once

#include <cstdint>

#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/types.hpp"

#include "types.hpp"

namespace igh::inline structs {

class vBGRu32x8
{
public:
    vBGRu32x8() : v_(_mm256_setzero_si256()), b_(0), g_(0), r_(0){};
    vBGRu32x8(const vBGRu32x8& rhs) = default;

    [[nodiscard]] uint32_t gray() const;

    static constexpr int CHANNELS = 3;

    v_u32x8 v_;
    uint32_t b_, g_, r_;
};

uint32_t vBGRu32x8::gray() const
{
    uint32_t b = b_;
    uint32_t g = g_;
    uint32_t r = r_;

    constexpr uint PIXELS = sizeof(v_u8x16) / vBGRu32x8::CHANNELS;
    uint32_t accbuf[sizeof(v_u32x8)];
    _mm256_storeu_epi16(accbuf, v_);

    const uint32_t* buf_cursor = accbuf;
    for (uint itmp = 0; itmp < PIXELS; itmp++) {
        b += *(buf_cursor++);
        g += *(buf_cursor++);
        r += *(buf_cursor++);
    }

    return b * ABLUE + g * AGREEN + r * ARED;
}

} // namespace igh::inline structs