#include <cstdint>

#include "imghash/helper/common/macros.h"
#include "types.hpp"

namespace igh::inline func {

static const v_u8x16 HEAD_MASKS[] = {
    _mm_bslli_si128(_mm_set1_epi32(-1), 0),  _mm_bslli_si128(_mm_set1_epi32(-1), 1),
    _mm_bslli_si128(_mm_set1_epi32(-1), 2),  _mm_bslli_si128(_mm_set1_epi32(-1), 3),
    _mm_bslli_si128(_mm_set1_epi32(-1), 4),  _mm_bslli_si128(_mm_set1_epi32(-1), 5),
    _mm_bslli_si128(_mm_set1_epi32(-1), 6),  _mm_bslli_si128(_mm_set1_epi32(-1), 7),
    _mm_bslli_si128(_mm_set1_epi32(-1), 8),  _mm_bslli_si128(_mm_set1_epi32(-1), 9),
    _mm_bslli_si128(_mm_set1_epi32(-1), 10), _mm_bslli_si128(_mm_set1_epi32(-1), 11),
    _mm_bslli_si128(_mm_set1_epi32(-1), 12), _mm_bslli_si128(_mm_set1_epi32(-1), 13),
    _mm_bslli_si128(_mm_set1_epi32(-1), 14), _mm_bslli_si128(_mm_set1_epi32(-1), 15),
};

static IGH_FORCEINLINE v_u8x16 get_head_mask(uint zeros) { return HEAD_MASKS[zeros]; }

static const v_u8x16 TAIL_MASKS[] = {
    _mm_bsrli_si128(_mm_set1_epi32(-1), 0),  _mm_bsrli_si128(_mm_set1_epi32(-1), 1),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 2),  _mm_bsrli_si128(_mm_set1_epi32(-1), 3),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 4),  _mm_bsrli_si128(_mm_set1_epi32(-1), 5),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 6),  _mm_bsrli_si128(_mm_set1_epi32(-1), 7),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 8),  _mm_bsrli_si128(_mm_set1_epi32(-1), 9),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 10), _mm_bsrli_si128(_mm_set1_epi32(-1), 11),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 12), _mm_bsrli_si128(_mm_set1_epi32(-1), 13),
    _mm_bsrli_si128(_mm_set1_epi32(-1), 14), _mm_bsrli_si128(_mm_set1_epi32(-1), 15),
};

static IGH_FORCEINLINE v_u8x16 get_tail_mask(uint zeros) { return TAIL_MASKS[zeros]; }

static IGH_FORCEINLINE uint32_t sum_u32x8_to_u32(v_u32x8 src)
{
    src = _mm256_hadd_epi32(src, src);

    alignas(32) uint32_t store[8];
    _mm256_store_si256((__m256i*)&store, src);

    uint32_t sum = store[0] + store[1] + store[4] + store[5];
    return sum;
}

static IGH_FORCEINLINE uint64_t sum_u32x8_to_u64(v_u32x8 src)
{
    src = _mm256_hadd_epi32(src, src);

    alignas(32) uint32_t store[8];
    _mm256_store_si256((__m256i*)&store, src);

    uint64_t sum = (uint64_t)store[0] + store[1] + store[4] + store[5];
    return sum;
}

} // namespace igh::inline func