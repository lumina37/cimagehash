#pragma once

#include <numeric>

#include "imghash/average_hash/consts.hpp"
#include "imghash/helper/avx2/func.hpp"
#include "imghash/helper/avx2/structs.hpp"
#include "imghash/helper/common/types.hpp"

namespace igh::ahash::inline avx2 {

static IGH_FORCEINLINE void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    const v_u32x8 average_i32x8 = _mm256_set1_epi32((int)average);
    const v_u32x8 permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    uint8_t* dst_cursor = dst;
    const uint32_t* hash_cursor = hash_buf;
    for (uint i = 0; i < 2; i++, dst_cursor += 4) {
        v_u16x16 hashj_i16x16[2];
        for (uint j = 0; j < 2; j++) {
            v_u32x8 hashk_i32x8[2];
            for (uint k = 0; k < 2; k++, hash_cursor += 8) {
                v_u32x8 hashcmp = _mm256_loadu_si256((v_u32x8*)hash_cursor);
                hashk_i32x8[k] = _mm256_cmpgt_epi32(hashcmp, average_i32x8);
            }
            hashj_i16x16[j] = _mm256_packs_epi32(hashk_i32x8[0], hashk_i32x8[1]);
        }
        v_u8x32 hashi_i8x32 = _mm256_packs_epi16(hashj_i16x16[0], hashj_i16x16[1]);
        hashi_i8x32 = _mm256_permutevar8x32_epi32(hashi_i8x32, permute_mask);
        uint32_t hashi_half = (uint32_t)_mm256_movemask_epi8(hashi_i8x32);
        memcpy(dst_cursor, &hashi_half, sizeof(hashi_half));
    }
}

static IGH_FORCEINLINE void _collect_ch3_seg(const uint8_t* row_start, const Segment& segment, vBlockSumU32* block_sum)
{
    const uint8_t* cursor = row_start + segment.shift;
    uint8_t targetv = segment.init_targetv;

    v_u8x16 head = _mm_load_si128((v_u8x16*)cursor);
    head = _mm_and_si128(head, segment.head_mask);
    v_u16x16 expand_head = _mm256_cvtepu8_epi16(head);
    block_sum->v_[targetv] = _mm256_add_epi16(expand_head, block_sum->v_[targetv]);
    targetv = Segment::LOOP_HELPER[targetv + 1];
    cursor += sizeof(v_u8x16);

    for (uint iloop = 0; iloop < segment.loop_num; iloop++) {
        v_u8x16 v = _mm_load_si128((v_u8x16*)cursor);
        v_u16x16 expand_v = _mm256_cvtepu8_epi16(v);
        block_sum->v_[targetv] = _mm256_add_epi16(expand_v, block_sum->v_[targetv]);
        targetv = Segment::LOOP_HELPER[targetv + 1];
        cursor += sizeof(v_u8x16);
    }

    v_u8x16 tail = _mm_load_si128((v_u8x16*)cursor);
    tail = _mm_and_si128(tail, segment.tail_mask);
    v_u16x16 expand_tail = _mm256_cvtepu8_epi16(tail);
    block_sum->v_[targetv] = _mm256_add_epi16(expand_tail, block_sum->v_[targetv]);
}

static IGH_FORCEINLINE void _collect_ch3_row(const uint8_t* row_start, const Segment (&segments)[DST_W],
                                             vBlockSumU32* block_sums)
{
    for (uint iblk = 0; iblk < DST_W; iblk++) {
        const auto& segment = segments[iblk];
        _collect_ch3_seg(row_start, segment, block_sums + iblk);
    }
}

static IGH_FORCEINLINE void compute_ch3_div8(const uint8_t* src, const uint width, const uint height,
                                             const uint row_step, uint8_t* dst)
{
    const uint block_row_num = height / DST_H; // how many rows in a block
    const uint block_width_bytes = width / DST_W * vBlockSumU32::CHANNELS;

    Segment segments[DST_W];
    for (uint i = 0; i < DST_W; i++) {
        segments[i] = Segment(i * block_width_bytes, block_width_bytes);
    }

    uint32_t resized_8x8[HASH_LEN];
    const uint8_t* src_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (uint idstrow = 0; idstrow < DST_H; idstrow++) {
        vBlockSumU32 block_sums[DST_W]{};
        for (uint isrcrow = 0; isrcrow < block_row_num; isrcrow++) {
            _collect_ch3_row(src_cursor, segments, block_sums);
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
    assert(width >= 64);
    assert(height >= 64);
    assert(align_le((size_t)src, sizeof(v_u8x16)) == (size_t)src);

    if (((width % DST_W) & (height % DST_H)) == 0) {
        compute_ch3_div8(src, width, height, row_step, dst);
    } else {
        dst = nullptr;
        return;
    }
}

} // namespace igh::ahash::inline avx2