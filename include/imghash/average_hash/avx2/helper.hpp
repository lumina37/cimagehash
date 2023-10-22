#include <cstdint>
#include <immintrin.h>

namespace igh::average {

static inline uint32_t sum_i32x8_to_u32(__m256i i32x8)
{
    i32x8 = _mm256_hadd_epi32(i32x8, i32x8);

    uint32_t store[8];
    _mm256_storeu_si256((__m256i*)&store, i32x8);

    uint32_t sum = store[0] + store[1] + store[4] + store[5];
    return sum;
}

static inline uint64_t sum_i32x8_to_u64(__m256i i32x8)
{
    i32x8 = _mm256_hadd_epi32(i32x8, i32x8);

    uint32_t store[8];
    _mm256_storeu_si256((__m256i*)&store, i32x8);

    uint64_t sum = (uint64_t)store[0] + store[1] + store[4] + store[5];
    return sum;
}

} // namespace igh::average