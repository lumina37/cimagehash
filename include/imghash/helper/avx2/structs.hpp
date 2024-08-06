#pragma once

#include "imghash/helper/avx2/types.hpp"
#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/macros.h"

namespace igh::inline structs {

class vBlockSumU32
{
public:
    [[nodiscard]] IGH_FORCEINLINE uint32_t gray() const;

    static constexpr uint CHANNELS = 3;

    v_u16x16 v_[3]{};
};

IGH_FORCEINLINE uint32_t vBlockSumU32::gray() const
{
    alignas(32) uint16_t bgr[sizeof(v_u16x16) * CHANNELS / sizeof(uint16_t)];

    _mm256_store_si256((__m256i*)&bgr + 0, v_[0]);
    _mm256_store_si256((__m256i*)&bgr + 1, v_[1]);
    _mm256_store_si256((__m256i*)&bgr + 2, v_[2]);

    uint32_t b = 0;
    uint32_t g = 0;
    uint32_t r = 0;

    for (uint i = 0; i < sizeof(v_u16x16) / sizeof(uint16_t); i++) {
        b += bgr[i * CHANNELS + 0];
        g += bgr[i * CHANNELS + 1];
        r += bgr[i * CHANNELS + 2];
    }

    uint32_t gray = b * ABLUE + g * AGREEN + r * ARED;
    return gray;
}

class Segment
{
public:
    Segment() = default;
    Segment(size_t start, size_t len);

    static constexpr size_t LOOP_HELPER[6] = {0, 1, 2, 0, 1, 2};

    v_u8x16 head_mask;
    v_u8x16 tail_mask;
    size_t shift;
    uint8_t init_targetv;
    uint8_t loop_num;
};

Segment::Segment(size_t start, size_t len)
{
    this->shift = align_le(start, sizeof(v_u8x16));
    this->init_targetv = (start / sizeof(v_u8x16)) % vBlockSumU32::CHANNELS;

    size_t aligned_start = align_ge(start, sizeof(v_u8x16));
    size_t head_len = aligned_start - start;
    head_mask = get_head_mask(sizeof(v_u8x16) - head_len);

    size_t res_len = len - head_len;
    loop_num = (res_len - 1) / sizeof(v_u8x16);

    size_t tail_len = res_len - loop_num * sizeof(v_u8x16);
    tail_mask = get_tail_mask(sizeof(v_u8x16) - tail_len);
}

} // namespace igh::inline structs