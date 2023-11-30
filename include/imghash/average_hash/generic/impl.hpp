#include <cassert>
#include <cstdint>
#include <intrin.h>
#include <numeric>

#include "imghash/helper/consts.hpp"
#include "imghash/helper/macros.h"
#include "imghash/helper/structs.hpp"
#include "imghash/helper/timer.hpp"
#include "imghash/helper/typedefs.h"

namespace igh::average {

#if 0

static IGH_FORCEINLINE void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    const __m256i average_i32x8 = _mm256_set1_epi32((int)average);
    static const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    uint8_t* dst_cursor = dst;
    const uint32_t* hash_cursor = hash_buf;
    for (int i = 0; i < 2; i++, dst_cursor += 4) {
        __m256i hashj_i16x16[2];
        for (int j = 0; j < 2; j++) {
            __m256i hashk_i32x8[2];
            for (int k = 0; k < 2; k++, hash_cursor += 8) {
                __m256i hashcmp = _mm256_loadu_si256((__m256i*)hash_cursor);
                hashk_i32x8[k] = _mm256_cmpgt_epi32(hashcmp, average_i32x8);
            }
            hashj_i16x16[j] = _mm256_packs_epi32(hashk_i32x8[0], hashk_i32x8[1]);
        }
        __m256i hashi_i8x32 = _mm256_packs_epi16(hashj_i16x16[0], hashj_i16x16[1]);
        hashi_i8x32 = _mm256_permutevar8x32_epi32(hashi_i8x32, permute_mask);
        uint32_t hashi_half = (uint32_t)_mm256_movemask_epi8(hashi_i8x32);
        memcpy(dst_cursor, &hashi_half, sizeof(hashi_half));
    }
}

#else

static IGH_FORCEINLINE void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    constexpr int GROUP_BITS = sizeof(uint8_t) * 8;
    constexpr int GROUPS = HASH_LEN / GROUP_BITS;

    const uint32_t* hash_cursor = hash_buf;
    uint8_t* dst_cursor = dst;

    for (int idstgp = 0; idstgp < GROUPS; idstgp++) {
        *dst_cursor = 0;
        uint8_t mask = 0b00000001;
        for (int ibit = 0; ibit < GROUP_BITS; ibit++, mask <<= 1, hash_cursor++) {
            *dst_cursor |= *hash_cursor > average ? mask : 0;
        }
        dst_cursor++;
    }
}

#endif

static IGH_FORCEINLINE void _collect_3ch_rowseg(const uint8_t* src, const int pixes, BGRu32* pBGR)
{
    const uint8_t* src_cursor = src;

    for (int ipix = 0; ipix < pixes; ipix++) {
        pBGR->b += *(src_cursor++);
        pBGR->g += *(src_cursor++);
        pBGR->r += *(src_cursor++);
    }
}

static IGH_FORCEINLINE void _collect_3ch_row(const uint8_t* src, const int width, BGRu32* pBGRs)
{
    const int block_cols = width / DST_W;
    const uint8_t* src_cursor = src;

    for (int iblk = 0; iblk < DST_W; iblk++, src_cursor += block_cols * BGRu32::lanes) {
        _collect_3ch_rowseg(src_cursor, block_cols, pBGRs + iblk);
    }
}

static IGH_FORCEINLINE void compute_ch3_div8(const uint8_t* src, const int width, const int height, const int row_step,
                                             uint8_t* dst)
{
    const int block_rows = height / DST_H;
    alignas(64) uint32_t resized_8x8[HASH_LEN];

    const uint8_t* src_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (int idstrow = 0; idstrow < DST_H; idstrow++) {
        BGRu32 BGRs[DST_W]{};
        for (int isrcrow = 0; isrcrow < block_rows; isrcrow++) {
            _collect_3ch_row(src_cursor, width, BGRs);
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

static void compute(uint8_t* src, int width, int height, int row_step, uint8_t* dst)
{
    assert(width >= 32);
    assert(height >= 32);

    compute_ch3_div8(src, width, height, row_step, dst);
}

} // namespace igh::average