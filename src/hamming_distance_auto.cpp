// SIMD popcount strategy used by the SSSE3/AVX/AVX2/AVX512 implementations below.
// For each vector chunk, compute `x = a ^ b` so set bits in `x` are differing bits.
// Each byte in `x` is split into low/high 4-bit nibbles, and each nibble (0..15)
// is mapped through a 16-entry popcount LUT via byte-wise shuffle instructions.
// Adding low+high lookup results gives per-byte bit counts (0..8); `sad_epu8` then
// sums those byte counts within each 64-bit lane, yielding one Hamming distance per uint64_t.

#include "hamming_distance_impl.hpp"

#include <bit>

namespace hamming_distance_impl
{
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
__attribute__((target_clones("arch=x86-64-v2,arch=x86-64-v3,arch=x86-64-v4,default")))
#endif
void auto_vectorized(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
#if defined(__clang__)
#    pragma clang loop vectorize(enable)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}
} // namespace hamming_distance_impl
