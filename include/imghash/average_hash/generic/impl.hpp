#pragma once

#include <cassert>
#include <numeric>

#include "imghash/helper/common/types.hpp"
#include "imghash/helper/generic/structs.hpp"
#include "imghash/average_hash/consts.hpp"

namespace igh::ahash::inline generic {

static IGH_FORCEINLINE void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    constexpr uint GROUP_BITS = sizeof(uint8_t) * 8;
    constexpr uint GROUPS = HASH_LEN / GROUP_BITS;

    const uint32_t* hash_cursor = hash_buf;
    uint8_t* dst_cursor = dst;

    for (uint idstgp = 0; idstgp < GROUPS; idstgp++) {
        *dst_cursor = 0;
        uint8_t mask = 0b00000001;
        for (uint ibit = 0; ibit < GROUP_BITS; ibit++) {
            *dst_cursor |= *hash_cursor > average ? mask : 0;
            mask <<= 1;
            hash_cursor++;
        }
        dst_cursor++;
    }
}

static IGH_FORCEINLINE void _collect_ch3_seg(const uint8_t* seg_start, const uint block_width, BlockSumU32* block_sum)
{
    const uint8_t* cursor = seg_start;
    for (uint ipix = 0; ipix < block_width; ipix++) {
        block_sum->b_ += *(cursor++);
        block_sum->g_ += *(cursor++);
        block_sum->r_ += *(cursor++);
    }
}

static IGH_FORCEINLINE void _collect_ch3_row(const uint8_t* row_start, const uint block_width, BlockSumU32* block_sums)
{
    const uint8_t* src_cursor = row_start;
    for (uint iblk = 0; iblk < DST_W; iblk++) {
        _collect_ch3_seg(src_cursor, block_width, block_sums + iblk);
        src_cursor += block_width * BlockSumU32::CHANNELS;
    }
}

static IGH_FORCEINLINE void compute_ch3_div8(const uint8_t* src, const uint width, const uint height,
                                             const uint row_step, uint8_t* dst)
{
    const uint block_row_num = height / DST_H; // how many rows in a block
    const uint block_width = width / DST_W;

    uint32_t resized_8x8[HASH_LEN];
    const uint8_t* src_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (uint idstrow = 0; idstrow < DST_H; idstrow++) {
        BlockSumU32 block_sums[DST_W]{};
        for (uint isrcrow = 0; isrcrow < block_row_num; isrcrow++) {
            _collect_ch3_row(src_cursor, block_width, block_sums);
            src_cursor += row_step;
        }
        for (const auto& block_sum : block_sums) {
            *dst_cursor = block_sum.gray();
            dst_cursor++;
        }
    }

    const uint64_t sum = std::accumulate(resized_8x8, resized_8x8 + HASH_LEN, (uint64_t)(HASH_LEN / 2));
    const uint32_t average = (uint32_t)(sum / HASH_LEN);

    /* Compare to the average and output average hash */
    _compute_hash(resized_8x8, average, dst);
}

static void compute(uint8_t* src, uint width, uint height, uint row_step, uint8_t* dst)
{
    assert(width >= 32);
    assert(height >= 32);

    compute_ch3_div8(src, width, height, row_step, dst);
}

} // namespace igh::ahash::inline generic