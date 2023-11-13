#include <cstdint>
#include <immintrin.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

static inline void _collect_rgb2x48(const uint8_t* row0, const uint8_t* row1, __m256i* b, __m256i* g, __m256i* r)
{
    __m128i x0, x1, x2, x3, x4, x5;
    __m128i y0, y1, y2, y3, y4, y5;
    x0 = _mm_loadu_si128((__m128i*)(row0 + 0));
    x1 = _mm_loadu_si128((__m128i*)(row0 + 16));
    x2 = _mm_loadu_si128((__m128i*)(row0 + 32));
    x3 = _mm_loadu_si128((__m128i*)(row1 + 0));
    x4 = _mm_loadu_si128((__m128i*)(row1 + 16));
    x5 = _mm_loadu_si128((__m128i*)(row1 + 32));

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

    __m256i b0 = _mm256_cvtepu8_epi16(x0);
    __m256i b1 = _mm256_cvtepu8_epi16(x3);
    __m256i g0 = _mm256_cvtepu8_epi16(x1);
    __m256i g1 = _mm256_cvtepu8_epi16(x4);
    __m256i r0 = _mm256_cvtepu8_epi16(x2);
    __m256i r1 = _mm256_cvtepu8_epi16(x5);

    *b = _mm256_add_epi16(b0, b1);
    *g = _mm256_add_epi16(g0, g1);
    *r = _mm256_add_epi16(r0, r1);
}

TEST_CASE("RGB", "[rgb]")
{
    uint8_t row0[48]{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                     25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
    uint8_t row1[48]{49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                     73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96};

    uint8_t bres[16]{};
    uint8_t gres[16]{};
    uint8_t rres[16]{};
    uint8_t barr[16];
    uint8_t garr[16];
    uint8_t rarr[16];
    __m256i b, g, r;

    _collect_rgb2x48(row0, row1, &b, &g, &r);

    _mm256_storeu_si256((__m256i*)barr, b);
    for (size_t i = 0; i < 32; i++) {
        REQUIRE(bres[i] == barr[i]);
    }
}

TEST_CASE("Average", "[average]")
{
    // Traditional Method
    uint32_t hash[64];
    uint32_t trad_acc = 0;
    for (uint32_t i = 0; i < 64; i++) {
        hash[i] = i;
        trad_acc += i;
    }
    double trad_average = (double)trad_acc / 64;

    // SIMD Method
    uint32_t* hash_cursor = hash;
    __m256i acc = _mm256_set_epi32(0, (int)hash_cursor[3], 0, (int)hash_cursor[2], 0, (int)hash_cursor[1], 0,
                                   (int)hash_cursor[0]); // init with first iter
    __m256i tmp;
    hash_cursor += 4;
    for (int iacc = 1; iacc < 16; iacc++, hash_cursor += 4) {
        tmp = _mm256_set_epi32(0, (int)hash_cursor[3], 0, (int)hash_cursor[2], 0, (int)hash_cursor[1], 0,
                               (int)hash_cursor[0]);
        acc = _mm256_add_epi64(acc, tmp);
    }

    uint64_t acc_buf[4];
    _mm256_storeu_si256((__m256i*)acc_buf, acc);
    double average = (double)(acc_buf[0] + acc_buf[1] + acc_buf[2] + acc_buf[3]) / 64;

    REQUIRE(average == trad_average);
}

TEST_CASE("Permute", "[permute]")
{
    int32_t a[8]{11, 12, 13, 14, 15, 16, 17, 18};
    __m256i a_i32x8 = _mm256_loadu_si256((__m256i*)a);
    int32_t b[8]{21, 22, 23, 24, 25, 26, 27, 28};
    __m256i b_i32x8 = _mm256_loadu_si256((__m256i*)b);
    int32_t c[8]{31, 32, 33, 34, 35, 36, 37, 38};
    __m256i c_i32x8 = _mm256_loadu_si256((__m256i*)c);
    int32_t d[8]{41, 42, 43, 44, 45, 46, 47, 48};
    __m256i d_i32x8 = _mm256_loadu_si256((__m256i*)d);

    __m256i ab_i16x16 = _mm256_packs_epi32(a_i32x8, b_i32x8);
    __m256i cd_i16x16 = _mm256_packs_epi32(c_i32x8, d_i32x8);

    __m256i abcd_i8x32 = _mm256_packs_epi16(ab_i16x16, cd_i16x16);

    __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i permute_res = _mm256_permutevar8x32_epi32(abcd_i8x32, permute_mask);

    uint8_t res[32]{11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                    31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48};
    uint8_t simd_res[32];
    _mm256_storeu_si256((__m256i*)simd_res, permute_res);

    for (size_t i = 0; i < 32; i++) {
        REQUIRE(simd_res[i] == res[i]);
    }
}
