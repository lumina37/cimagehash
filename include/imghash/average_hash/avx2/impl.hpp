#include <intrin.h>
#include <numeric>

#include "imghash/helper/avx2/func.hpp"
#include "imghash/helper/avx2/structs.hpp"
#include "imghash/helper/common/consts.hpp"
#include "imghash/helper/common/macros.h"
#include "imghash/helper/common/timer.hpp"
#include "imghash/helper/common/types.hpp"

namespace igh::ahash::inline avx2 {

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

constexpr uint CHANNELS = 3;
constexpr uint COL_UNIT_NPIX = 16;
constexpr uint COL_UNIT_NBYTE = COL_UNIT_NPIX * CHANNELS;

/**
 *
 * @note Each unit is in size of 2(lanes)x16 pixels
 */
static IGH_FORCEINLINE void _collect_3ch_unit(const uint8_t* lane0, const uint8_t* lane1, v_u16x16* dst_b,
                                              v_u16x16* dst_g, v_u16x16* dst_r)
{
    v_u8x16 x0, x1, x2, x3, x4, x5;
    v_u8x16 y0, y1, y2, y3, y4, y5;
    x0 = _mm_loadu_si128((__m128i*)(lane0 + 0 * COL_UNIT_NPIX));
    x1 = _mm_loadu_si128((__m128i*)(lane0 + 1 * COL_UNIT_NPIX));
    x2 = _mm_loadu_si128((__m128i*)(lane0 + 2 * COL_UNIT_NPIX));
    x3 = _mm_loadu_si128((__m128i*)(lane1 + 0 * COL_UNIT_NPIX));
    x4 = _mm_loadu_si128((__m128i*)(lane1 + 1 * COL_UNIT_NPIX));
    x5 = _mm_loadu_si128((__m128i*)(lane1 + 2 * COL_UNIT_NPIX));

    y0 = _mm_unpacklo_epi8(x0, x3);
    y1 = _mm_unpackhi_epi8(x0, x3);
    y2 = _mm_unpacklo_epi8(x1, x4);
    y3 = _mm_unpackhi_epi8(x1, x4);
    y4 = _mm_unpacklo_epi8(x2, x5);
    y5 = _mm_unpackhi_epi8(x2, x5);

    x0 = _mm_unpacklo_epi8(y0, y3);
    x1 = _mm_unpackhi_epi8(y0, y3);
    x2 = _mm_unpacklo_epi8(y1, y4);
    x3 = _mm_unpackhi_epi8(y1, y4);
    x4 = _mm_unpacklo_epi8(y2, y5);
    x5 = _mm_unpackhi_epi8(y2, y5);

    y0 = _mm_unpacklo_epi8(x0, x3);
    y1 = _mm_unpackhi_epi8(x0, x3);
    y2 = _mm_unpacklo_epi8(x1, x4);
    y3 = _mm_unpackhi_epi8(x1, x4);
    y4 = _mm_unpacklo_epi8(x2, x5);
    y5 = _mm_unpackhi_epi8(x2, x5);

    x0 = _mm_unpacklo_epi8(y0, y3);
    x1 = _mm_unpackhi_epi8(y0, y3);
    x2 = _mm_unpacklo_epi8(y1, y4);
    x3 = _mm_unpackhi_epi8(y1, y4);
    x4 = _mm_unpacklo_epi8(y2, y5);
    x5 = _mm_unpackhi_epi8(y2, y5);

    v_u16x16 b0 = _mm256_cvtepu8_epi16(x0);
    v_u16x16 b1 = _mm256_cvtepu8_epi16(x3);
    v_u16x16 g0 = _mm256_cvtepu8_epi16(x1);
    v_u16x16 g1 = _mm256_cvtepu8_epi16(x4);
    v_u16x16 r0 = _mm256_cvtepu8_epi16(x2);
    v_u16x16 r1 = _mm256_cvtepu8_epi16(x5);

    *dst_b = _mm256_add_epi16(b0, b1);
    *dst_g = _mm256_add_epi16(g0, g1);
    *dst_r = _mm256_add_epi16(r0, r1);
}

/**
 *
 * @note Each block will form a pixel in resized image
 */
static IGH_FORCEINLINE void _collect_3ch_block(const uint8_t* src, const uint rows, const uint cols,
                                               const uint row_step, uint32_t* dst)
{
    /* if you sum up too many units, uint16 will get overflowed */
    constexpr uint UNIT_LANES = 2;
    constexpr uint UNIT_ACCUMULATE_LIMIT =
        std::numeric_limits<uint16_t>::max() / std::numeric_limits<uint8_t>::max() / UNIT_LANES;
    constexpr uint COL_CHUNK_NUNIT = UNIT_ACCUMULATE_LIMIT;
    constexpr uint COL_CHUNK_NPIX = COL_CHUNK_NUNIT * COL_UNIT_NPIX;
    constexpr uint COL_CHUNK_NBYTE = COL_CHUNK_NUNIT * COL_UNIT_NBYTE;

    const uint row_pairs = rows / UNIT_LANES;
    const bool has_resi_row = (bool)(rows - row_pairs * UNIT_LANES);
    const uint col_chunks = cols / COL_CHUNK_NPIX;
    const uint col_chunk_resi_pix = cols - col_chunks * COL_CHUNK_NPIX;
    const uint col_resi_units = col_chunk_resi_pix / COL_UNIT_NPIX;
    const uint col_unit_resi_pix = col_chunk_resi_pix - col_resi_units * COL_UNIT_NPIX;

    auto _collect_rgb_chunk = [=](const uint8_t* chunk0, const uint8_t* chunk1, v_u32x8* cdst_b, v_u32x8* cdst_g,
                                  v_u32x8* cdst_r, const uint units) {
        v_u16x16 chunk_b = _mm256_setzero_si256();
        v_u16x16 chunk_g = _mm256_setzero_si256();
        v_u16x16 chunk_r = _mm256_setzero_si256();

        for (uint iunit = 0; iunit < units; iunit++, chunk0 += COL_UNIT_NBYTE, chunk1 += COL_UNIT_NBYTE) {
            v_u16x16 vb, vg, vr;
            _collect_3ch_unit(chunk0, chunk1, &vb, &vg, &vr);
            chunk_b = _mm256_add_epi16(chunk_b, vb);
            chunk_g = _mm256_add_epi16(chunk_g, vg);
            chunk_r = _mm256_add_epi16(chunk_r, vr);
        }

        v_u32x8 b0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_b, 0));
        v_u32x8 b1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_b, 1));
        *cdst_b = _mm256_add_epi32(b0, b1);

        v_u32x8 g0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_g, 0));
        v_u32x8 g1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_g, 1));
        *cdst_g = _mm256_add_epi32(g0, g1);

        v_u32x8 r0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_r, 0));
        v_u32x8 r1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(chunk_r, 1));
        *cdst_r = _mm256_add_epi32(r0, r1);
    };

    /* Collect row-pairs for each block */
    const uint8_t* row0_cursor = src;
    const uint8_t* row1_cursor = src + row_step;
    v_u32x8 rpair_b = _mm256_setzero_si256();
    v_u32x8 rpair_g = _mm256_setzero_si256();
    v_u32x8 rpair_r = _mm256_setzero_si256();
    v_u32x8 vb, vg, vr;
    uint32_t rpair_sb = 0;
    uint32_t rpair_sg = 0;
    uint32_t rpair_sr = 0;

    for (uint irpair = 0; irpair < row_pairs;
         irpair++, row0_cursor += row_step * UNIT_LANES, row1_cursor += row_step * UNIT_LANES) {

        /* Collect chunks for each row-pair */
        const uint8_t* chunk0_cursor = row0_cursor;
        const uint8_t* chunk1_cursor = row1_cursor;

        for (uint ichunk = 0; ichunk < col_chunks;
             ichunk++, chunk0_cursor += COL_CHUNK_NBYTE, chunk1_cursor += COL_CHUNK_NBYTE) {
            _collect_rgb_chunk(chunk0_cursor, chunk1_cursor, &vb, &vg, &vr, COL_CHUNK_NUNIT);
            rpair_b = _mm256_add_epi32(rpair_b, vb);
            rpair_g = _mm256_add_epi32(rpair_g, vg);
            rpair_r = _mm256_add_epi32(rpair_r, vr);
        }

        /* Collect the `col_resi_units` for each row-pair */
        _collect_rgb_chunk(chunk0_cursor, chunk1_cursor, &vb, &vg, &vr, col_resi_units);
        rpair_b = _mm256_add_epi32(rpair_b, vb);
        rpair_g = _mm256_add_epi32(rpair_g, vg);
        rpair_r = _mm256_add_epi32(rpair_r, vr);
        chunk0_cursor += COL_UNIT_NBYTE * col_resi_units;
        chunk1_cursor += COL_UNIT_NBYTE * col_resi_units;

        /* Collect the `unit_resi_pix` for each row-pair */
        for (uint iuresi = 0; iuresi < col_unit_resi_pix;
             iuresi += CHANNELS, chunk0_cursor += CHANNELS, chunk1_cursor += CHANNELS) {
            rpair_sb += (uint32_t)(chunk0_cursor[0]);
            rpair_sg += (uint32_t)(chunk0_cursor[1]);
            rpair_sr += (uint32_t)(chunk0_cursor[2]);
            rpair_sb += (uint32_t)(chunk1_cursor[0]);
            rpair_sg += (uint32_t)(chunk1_cursor[1]);
            rpair_sr += (uint32_t)(chunk1_cursor[2]);
        }
    }

    uint32_t dstb, dstg, dstr;
    if (!has_resi_row) {
        dstb = rpair_sb + sum_u32x8_to_u32(rpair_b);
        dstg = rpair_sg + sum_u32x8_to_u32(rpair_g);
        dstr = rpair_sr + sum_u32x8_to_u32(rpair_r);
    } else {
        /* Collect the `has_resi_row` for each block */
        const uint unit_pairs = cols / (COL_UNIT_NPIX * 2);
        const uint resirow_resi_pix = cols - unit_pairs * (COL_UNIT_NPIX * 2);
        const uint8_t* resirow_cursor = row1_cursor;

        v_u32x8 resirow_b, resirow_g, resirow_r;
        uint32_t rrowresi_sb = 0;
        uint32_t rrowresi_sg = 0;
        uint32_t rrowresi_sr = 0;

        _collect_rgb_chunk(resirow_cursor, resirow_cursor + COL_UNIT_NBYTE * unit_pairs, &resirow_b, &resirow_g,
                           &resirow_r, unit_pairs);

        resirow_cursor += COL_UNIT_NBYTE * unit_pairs * 2;
        for (uint irrowresi = 0; irrowresi < resirow_resi_pix; irrowresi++, resirow_cursor += CHANNELS) {
            rrowresi_sb += (uint32_t)((resirow_cursor + 0)[0]);
            rrowresi_sg += (uint32_t)((resirow_cursor + 1)[0]);
            rrowresi_sr += (uint32_t)((resirow_cursor + 2)[0]);
        }

        dstb = rpair_sb + rrowresi_sb + sum_u32x8_to_u32(_mm256_add_epi32(rpair_b, resirow_b));
        dstg = rpair_sg + rrowresi_sg + sum_u32x8_to_u32(_mm256_add_epi32(rpair_g, resirow_g));
        dstr = rpair_sb + rrowresi_sr + sum_u32x8_to_u32(_mm256_add_epi32(rpair_r, resirow_r));
    }

    *dst = dstb * ABLUE + dstg * AGREEN + dstr * ARED;
}

static IGH_FORCEINLINE void compute_ch3_div8(uint8_t* src, const uint width, const uint height, const uint row_step,
                                             uint8_t* dst)
{

    const uint block_cols = width / DST_W;
    const uint block_rows = height / DST_H;
    alignas(64) uint32_t resized_8x8[HASH_LEN];

    /* Resize and grayscale each row */
    uint8_t* row_block_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (uint idstrow = 0; idstrow < DST_W; idstrow++, row_block_cursor += block_rows * row_step) {
        uint8_t* col_block_cursor = row_block_cursor;
        for (uint idstcol = 0; idstcol < DST_W; idstcol++, col_block_cursor += block_cols * CHANNELS, dst_cursor++) {
            _collect_3ch_block(col_block_cursor, block_rows, block_cols, row_step, dst_cursor);
        }
    }

    /* Resize each col and compute the average value */
    uint32_t* rowdst_cursor = resized_8x8;
    v_u32x8 sum_u32x8 = _mm256_setzero_si256();
    v_u32x8 temp;
    for (uint idstrow = 0; idstrow < DST_H; idstrow++, rowdst_cursor += DST_W) {
        temp = _mm256_loadu_si256((__m256i*)rowdst_cursor); // init with first cycle
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, temp);
    }

    uint64_t sum_for_average = sum_u32x8_to_u64(sum_u32x8);
    const uint32_t average = (uint32_t)(sum_for_average / HASH_LEN);

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

} // namespace igh::ahash::inline avx2