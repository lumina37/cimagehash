#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include "const.hpp"
#include "helper.hpp"

namespace igh::average {

static inline __m256i _gather_24(const uint8_t* data)
{
    static const __m256i weight1 =
        _mm256_set_epi32(AGREEN, ABLUE, ARED, AGREEN, ABLUE, ARED, AGREEN, ABLUE); // lo [B G R B G R B G] hi
    static const __m256i weight2 =
        _mm256_set_epi32(ABLUE, ARED, AGREEN, ABLUE, ARED, AGREEN, ABLUE, ARED); // lo [R B G R B G R B] hi
    static const __m256i weight3 =
        _mm256_set_epi32(ARED, AGREEN, ABLUE, ARED, AGREEN, ABLUE, ARED, AGREEN); // lo [G R B G R B G R] hi

    __m256i pix_i32x8, acc_i32x8;

    pix_i32x8 = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)data));
    acc_i32x8 = _mm256_mullo_epi32(pix_i32x8, weight1);

    pix_i32x8 = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(data + 8)));
    pix_i32x8 = _mm256_mullo_epi32(pix_i32x8, weight2);
    acc_i32x8 = _mm256_add_epi32(acc_i32x8, pix_i32x8);

    pix_i32x8 = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(data + 16)));
    pix_i32x8 = _mm256_mullo_epi32(pix_i32x8, weight3);
    acc_i32x8 = _mm256_add_epi32(acc_i32x8, pix_i32x8);

    return acc_i32x8;
}

static inline void _compute_hash(const uint32_t* hash_buf, const uint32_t average, uint8_t* dst)
{
    const __m256i average_i32x8 = _mm256_set1_epi32((int)average);
    static const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    uint8_t* dst_cursor = dst;
    const uint32_t* hash_cursor = hash_buf;
    for (int i = 0; i < 2; i++) {
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
        uint32_t hashi_half = _mm256_movemask_epi8(hashi_i8x32);
        for (int idst = 0; idst < 4; idst++, dst_cursor++) {
            *dst_cursor = (uint8_t)(hashi_half >> (idst * 8)) & 0xFF;
        }
    }
}

static inline void compute_ch3_div8(uint8_t* src, int width, int height, int row_step, uint8_t* dst)
{
    constexpr int CHANNELS = 3;
    constexpr int COL_GROUP_PIXLEN = 8;
    constexpr int COL_GROUP_BYTELEN = COL_GROUP_PIXLEN * CHANNELS;

    const int col_each_from_npixs = width / DST_W;
    const int col_each_from_ngroups = col_each_from_npixs / COL_GROUP_PIXLEN;
    const int col_residual_pixs = col_each_from_npixs - col_each_from_ngroups * COL_GROUP_PIXLEN;

    /* Resize and grayscale each row */
    uint8_t* row_cursor = src;
    for (int irow = 0; irow < height; irow++, row_cursor += row_step) {
        __m256i tmp_g24;
        uint8_t* group_cursor = row_cursor;
        for (int idstcol = 0; idstcol < DST_W; idstcol++) {
            __m256i gather_i32x8 = _gather_24(group_cursor); // init with first cycle
            group_cursor += COL_GROUP_BYTELEN;
            for (int gp = 1; gp < col_each_from_ngroups; gp++, group_cursor += COL_GROUP_BYTELEN) {
                tmp_g24 = _gather_24(group_cursor);
                gather_i32x8 = _mm256_add_epi32(gather_i32x8, tmp_g24);
            }

            uint32_t sum = sum_i32x8_to_u32(gather_i32x8);
            for (int iresi = 0; iresi < col_residual_pixs; iresi++, group_cursor += CHANNELS) {
                sum += ABLUE * group_cursor[0] + AGREEN * group_cursor[1] + ARED * group_cursor[2];
            }

            ((uint32_t*)row_cursor)[idstcol] = sum; // EVIL memory reuse
        }
    }

    /* Resize each col and compute the average value */
    uint32_t resized_gray_img_8x8[HASH_LEN];
    uint32_t* rowdst_cursor = resized_gray_img_8x8;

    const int row_each_from_npixs = height / DST_H;

    uint64_t sum_for_average = 0;

    row_cursor = src;
    for (int idstrow = 0; idstrow < DST_H; idstrow++, rowdst_cursor += DST_W) {
        __m256i sum_i32x8 = _mm256_loadu_si256((__m256i*)row_cursor); // init with first cycle
        row_cursor += row_step;

        __m256i tmp;
        for (int irowpix = 1; irowpix < row_each_from_npixs; irowpix++, row_cursor += row_step) {
            tmp = _mm256_loadu_si256((__m256i*)row_cursor);
            sum_i32x8 = _mm256_add_epi32(sum_i32x8, tmp);
        }

        sum_for_average += sum_i32x8_to_u64(sum_i32x8);
        _mm256_storeu_si256((__m256i*)rowdst_cursor, sum_i32x8);
    }

    const uint32_t average = (uint32_t)((sum_for_average) / HASH_LEN);

    /* Compare to the average and output average hash */
    _compute_hash(resized_gray_img_8x8, average, dst);
}

static void compute(uint8_t* src, int width, int height, int row_step, uint8_t* dst)
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