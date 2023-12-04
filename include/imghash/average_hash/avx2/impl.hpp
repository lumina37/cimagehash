#include <intrin.h>
#include <numeric>

#include "imghash/helper/avx2/func.hpp"
#include "imghash/helper/avx2/structs.hpp"
#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/macros.h"
#include "imghash/helper/common/timer.hpp"
#include "imghash/helper/common/types.hpp"

namespace igh::average {

static IGH_FORCEINLINE void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    const v_u32x8 average_i32x8 = _mm256_set1_epi32((int)average);
    static const v_u32x8 permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

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

static IGH_FORCEINLINE void _collect_ch3_seg(const uint8_t* src, const uint pixes, vBGRu32x8* pBGR)
{
    constexpr uint GROUP_PIXS = sizeof(v_u8x16);
    const uint groups = pixes / GROUP_PIXS;
    const uint group_resi_pix = pixes - groups * GROUP_PIXS;
    const v_u16x16 zero = _mm256_setzero_si256();

    const uint8_t* src_cursor = src;
    for (uint igp = 0; igp < groups; igp++) {
        v_u16x16 b_b = _mm256_cvtepu8_epi16(_mm_loadu_si128((v_u8x16*)(src_cursor + 0 * sizeof(v_u8x16))));
        v_u32x8 bgrbgrbg_0 = _mm256_unpacklo_epi16(b_b, zero);
        v_u32x8 rbgrbgrg_0 = _mm256_unpackhi_epi16(b_b, zero);

        v_u16x16 g_g = _mm256_cvtepu8_epi16(_mm_loadu_si128((v_u8x16*)(src_cursor + 1 * sizeof(v_u8x16))));
        v_u32x8 grbgrbgr_0 = _mm256_unpacklo_epi16(g_g, zero);
        v_u32x8 bgrbgrbg_1 = _mm256_unpackhi_epi16(g_g, zero);

        v_u16x16 r_r = _mm256_cvtepu8_epi16(_mm_loadu_si128((v_u8x16*)(src_cursor + 2 * sizeof(v_u8x16))));
        v_u32x8 rbgrbgrg_1 = _mm256_unpacklo_epi16(r_r, zero);
        v_u32x8 grbgrbgr_1 = _mm256_unpackhi_epi16(r_r, zero);

        pBGR->vb_ = _mm256_add_epi32(pBGR->vb_, bgrbgrbg_0);
        pBGR->vb_ = _mm256_add_epi32(pBGR->vb_, bgrbgrbg_1);

        pBGR->vg_ = _mm256_add_epi32(pBGR->vg_, grbgrbgr_0);
        pBGR->vg_ = _mm256_add_epi32(pBGR->vg_, grbgrbgr_1);

        pBGR->vr_ = _mm256_add_epi32(pBGR->vr_, rbgrbgrg_0);
        pBGR->vr_ = _mm256_add_epi32(pBGR->vr_, rbgrbgrg_1);

        src_cursor += sizeof(v_u8x16) * vBGRu32x8::CHANNELS;
    }

    for (uint ipix = 0; ipix < group_resi_pix; ipix++) {
        pBGR->b_ += *(src_cursor++);
        pBGR->g_ += *(src_cursor++);
        pBGR->r_ += *(src_cursor++);
    }
}

static IGH_FORCEINLINE void _collect_ch3_row(const uint8_t* src, const uint width, vBGRu32x8* pBGRs)
{
    const uint block_cols = width / DST_W;
    const uint8_t* src_cursor = src;

    for (uint iblk = 0; iblk < DST_W; iblk++) {
        _collect_ch3_seg(src_cursor, block_cols, pBGRs + iblk);
        src_cursor += block_cols * vBGRu32x8::CHANNELS;
    }
}

static IGH_FORCEINLINE void compute_ch3_div8(const uint8_t* src, const uint width, const uint height,
                                             const uint row_step, uint8_t* dst)
{
    const uint block_rows = height / DST_H;
    alignas(sizeof(v_u32x8)) uint32_t resized_8x8[HASH_LEN];

    const uint8_t* src_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (uint idstrow = 0; idstrow < DST_H; idstrow++) {
        alignas(sizeof(v_u32x8)) vBGRu32x8 BGRs[DST_W]{};
        for (uint isrcrow = 0; isrcrow < block_rows; isrcrow++) {
            _collect_ch3_row(src_cursor, width, BGRs);
            src_cursor += row_step;
        }
        for (const auto& BGR : BGRs) {
            *dst_cursor = BGR.gray();
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

    if (((width % DST_W) & (height % DST_H)) == 0) {
        compute_ch3_div8(src, width, height, row_step, dst);
    } else {
        dst = nullptr;
        return;
    }
}

} // namespace igh::average