#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "imghash/average_hash/avx2/impl.hpp"

TEST_CASE("acc24", "[acc24]")
{
    static const uint8_t arr1[24]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    REQUIRE(abs((int64_t)igh::average::sum_i32x8_to_u32(igh::average::_gather_24(arr1)) - 1525728) < 4);

    static const uint8_t arr2[24]{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    REQUIRE(abs((int64_t)igh::average::sum_i32x8_to_u32(igh::average::_gather_24(arr2)) - 33423360) < 4);
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
