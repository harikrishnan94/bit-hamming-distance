#include "hamming_distance.hpp"

#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>


void hamming_distance(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b, std::span<std::uint64_t> results)
{
    if (a.size() != b.size() || a.size() != results.size())
    {
        throw std::invalid_argument{"Input spans must have the same size."};
    }

    const auto size = b.size();
    for (std::size_t i = 0; i < size; ++i)
    {
        const auto x = a[i] ^ b[i];
#if defined(__clang__) || defined(__GNUC__) || defined(_MSC_VER) && defined(_M_X64)
        results[i] = static_cast<std::uint64_t>(std::popcount(x));
#else
        // SWAR(SIMD - within - a - register) popcount fallback.
        auto v = x;
        v = v - ((v >> 1U) & 0x5555555555555555ULL);
        v = (v & 0x3333333333333333ULL) + ((v >> 2U) & 0x3333333333333333ULL);
        v = (v + (v >> 4U)) & 0x0F0F0F0F0F0F0F0FULL;
        results[i] = (v * 0x0101010101010101ULL) >> 56U;
#endif
    }
}
