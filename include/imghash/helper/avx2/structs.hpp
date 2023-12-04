#pragma once

#include <cstdint>

#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/types.hpp"

#include "types.hpp"

namespace igh::inline structs {

class vBGRu32x8
{
public:
    vBGRu32x8()
        : vb_(_mm256_setzero_si256()), vg_(_mm256_setzero_si256()), vr_(_mm256_setzero_si256()), b_(0), g_(0), r_(0){};
    vBGRu32x8(const vBGRu32x8& rhs) = default;

    [[nodiscard]] uint32_t gray() const;

    static constexpr int CHANNELS = 3;

    v_u32x8 vb_, vg_, vr_; // starts with B/G/R
    uint32_t b_, g_, r_;
};

uint32_t vBGRu32x8::gray() const
{
    constexpr uint NUMS_PER_V = sizeof(v_u32x8) / sizeof(uint32_t);
    constexpr uint ACCBUF_LEN = NUMS_PER_V * CHANNELS;
    constexpr uint PIXELS = ACCBUF_LEN / CHANNELS;
    uint32_t accbuf[ACCBUF_LEN];
    _mm256_storeu_epi32(((uint32_t*)accbuf + 0 * NUMS_PER_V), vb_);
    _mm256_storeu_epi32(((uint32_t*)accbuf + 1 * NUMS_PER_V), vg_);
    _mm256_storeu_epi32(((uint32_t*)accbuf + 2 * NUMS_PER_V), vr_);

    uint32_t b = b_;
    uint32_t g = g_;
    uint32_t r = r_;
    const uint32_t* buf_cursor = accbuf;
    for (uint itmp = 0; itmp < PIXELS; itmp++) {
        b += *(buf_cursor++);
        g += *(buf_cursor++);
        r += *(buf_cursor++);
    }

    return b * ABLUE + g * AGREEN + r * ARED;
}

} // namespace igh::inline structs