#include <cstdint>
#include <intrin.h>

#include "imghash/helper/typedefs.h"

namespace igh::average {

static inline uint32_t sum_u32x8_to_u32(v_u32x8 src)
{
    src = _mm256_hadd_epi32(src, src);

    uint32_t store[8];
    _mm256_storeu_si256((__m256i*)&store, src);

    uint32_t sum = store[0] + store[1] + store[4] + store[5];
    return sum;
}

static inline uint64_t sum_u32x8_to_u64(v_u32x8 src)
{
    src = _mm256_hadd_epi32(src, src);

    uint32_t store[8];
    _mm256_storeu_si256((__m256i*)&store, src);

    uint64_t sum = (uint64_t)store[0] + store[1] + store[4] + store[5];
    return sum;
}

} // namespace igh::average