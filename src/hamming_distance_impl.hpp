// SIMD popcount strategy used by the SSSE3/AVX/AVX2/AVX512 implementations below.
// For each vector chunk, compute `x = a ^ b` so set bits in `x` are differing bits.
// Each byte in `x` is split into low/high 4-bit nibbles, and each nibble (0..15)
// is mapped through a 16-entry popcount LUT via byte-wise shuffle instructions.
// Adding low+high lookup results gives per-byte bit counts (0..8); `sad_epu8` then
// sums those byte counts within each 64-bit lane, yielding one Hamming distance per uint64_t.

#pragma once

#include <cstddef>
#include <cstdint>

namespace hamming_distance_impl
{
void scalar(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size);
void auto_vectorized(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size);
void avx2(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size);
void avx512(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size);
} // namespace hamming_distance_impl
