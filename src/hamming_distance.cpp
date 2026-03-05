// SIMD popcount strategy used by the SSSE3/AVX/AVX2/AVX512 implementations below.
// For each vector chunk, compute `x = a ^ b` so set bits in `x` are differing bits.
// Each byte in `x` is split into low/high 4-bit nibbles, and each nibble (0..15)
// is mapped through a 16-entry popcount LUT via byte-wise shuffle instructions.
// Adding low+high lookup results gives per-byte bit counts (0..8); `sad_epu8` then
// sums those byte counts within each 64-bit lane, yielding one Hamming distance per uint64_t.

#include "hamming_distance.hpp"

#include "hamming_distance_impl.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>

namespace
{
using HammingDistanceImpl = void (*)(const std::uint64_t *, const std::uint64_t *, std::uint64_t *, std::size_t);

std::string normalize_env_selector_value(const char * value)
{
    if (value == nullptr)
    {
        return {};
    }

    std::string normalized;
    for (const unsigned char c : std::string{value})
    {
        if (std::isalnum(c) != 0)
        {
            normalized.push_back(static_cast<char>(std::tolower(c)));
        }
    }

    return normalized;
}

HammingDistanceImpl select_active_impl()
{
    const char * implementation_env_value = std::getenv("HAMMING_DISTANCE_IMPLEMENTATION");
    if (implementation_env_value != nullptr)
    {
        const std::string normalized = normalize_env_selector_value(implementation_env_value);

        if (normalized == "scalar")
        {
            return &hamming_distance_impl::scalar;
        }
        if (normalized == "auto" || normalized == "autovectorized" || normalized == "autovectorization")
        {
            return &hamming_distance_impl::auto_vectorized;
        }
        if (normalized == "avx2")
        {
            return &hamming_distance_impl::avx2;
        }
        if (normalized == "avx512" || normalized == "avx512f")
        {
            return &hamming_distance_impl::avx512;
        }

        throw std::invalid_argument{
            "Unsupported value for HAMMING_DISTANCE_IMPLEMENTATION: '" + std::string{implementation_env_value}
            + "'. Supported values: scalar, auto, avx2, avx512."};
    }

    return &hamming_distance_impl::auto_vectorized;
}
} // namespace

void hamming_distance(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b, std::span<std::uint64_t> results)
{
    if (a.size() != b.size() || a.size() != results.size())
    {
        throw std::invalid_argument{"Input spans must have the same size."};
    }

    const auto * a_ptr = a.data();
    const auto * b_ptr = b.data();
    auto * results_ptr = results.data();

    static const HammingDistanceImpl implementation = select_active_impl();
    implementation(a_ptr, b_ptr, results_ptr, a.size());
}
