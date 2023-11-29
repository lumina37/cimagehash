#pragma once

namespace igh {

template <typename T> consteval T cround(T value)
{
    T floor_v = static_cast<T>(static_cast<int64_t>(value));
    T diff = value - floor_v;
    if (diff > 0.5) {
        return static_cast<T>(floor_v + 1);
    } else {
        return static_cast<T>(floor_v);
    }
}

} // namespace igh