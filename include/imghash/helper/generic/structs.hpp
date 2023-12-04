#pragma once

#include "imghash/helper/common/consts.hpp"

namespace igh::inline structs {

template <typename T> class BGR
{
public:
    BGR() = default;
    BGR(const BGR& rhs) = default;

    [[nodiscard]] T gray() const;

    T b_, g_, r_;
};

template <typename T> T BGR<T>::gray() const { return b_ * ABLUE + g_ * AGREEN + r_ * ARED; }

using BGRu32 = BGR<uint32_t>;

} // namespace igh::inline structs