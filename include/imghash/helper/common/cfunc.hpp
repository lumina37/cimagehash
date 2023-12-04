#pragma once

namespace igh::inline cfunc {

template <typename T> constexpr T cround(T value)
{
    T floor_v = static_cast<T>(static_cast<int64_t>(value));
    T diff = value - floor_v;
    if (diff > 0.5) {
        return static_cast<T>(floor_v + 1);
    } else {
        return static_cast<T>(floor_v);
    }
}

template <typename Tv, typename Ta> constexpr Tv align_to(const Tv addr, const Ta align)
{
    return (addr + align - 1) & ~(align - 1);
}

} // namespace igh::inline cfunc