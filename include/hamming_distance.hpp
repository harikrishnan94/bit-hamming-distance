#pragma once

#include <bit>
#include <cstdint>


inline std::uint64_t hamming_distance(std::uint64_t a, std::uint64_t b)
{
    return std::popcount(a ^ b);
}
