#include "imghash/helper/consts.hpp"
#include "imghash/helper/macros.h"
#include "imghash/helper/structs.hpp"
#include "imghash/helper/timer.hpp"
#include "imghash/helper/typedefs.h"

namespace igh::average {

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

static IGH_FORCEINLINE void compute_ch3_div8(uint8_t* src, const int width, const int height, const int row_step,
                                             uint8_t* dst)
{
    const int block_rows = height / DST_H;
    alignas(64) uint32_t resized_8x8[HASH_LEN];

    /* Resize and grayscale each row */
    uint8_t* src_cursor = src;
    uint32_t* dst_cursor = resized_8x8;
    for (int idstrow = 0; idstrow < DST_H; idstrow++) {
        BGRu32 BGRs[DST_W]{};
        for (int isrcrow = 0; isrcrow < block_rows; isrcrow++, src_cursor += row_step) {
            _collect_3ch_row(src_cursor, width, BGRs);
        }
        for (const auto& BGR : BGRs) {
            *(dst_cursor++) = BGR.gray();
        }
    }

    uint64_t sum_for_average = 0;
    for (const uint32_t val : resized_8x8) {
        sum_for_average += val;
    }
    const uint32_t average = (uint32_t)(sum_for_average / HASH_LEN);

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