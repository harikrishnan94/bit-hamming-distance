// SIMD popcount strategy used by the SSSE3/AVX/AVX2/AVX512 implementations below.
// For each vector chunk, compute `x = a ^ b` so set bits in `x` are differing bits.
// Each byte in `x` is split into low/high 4-bit nibbles, and each nibble (0..15)
// is mapped through a 16-entry popcount LUT via byte-wise shuffle instructions.
// Adding low+high lookup results gives per-byte bit counts (0..8); `sad_epu8` then
// sums those byte counts within each 64-bit lane, yielding one Hamming distance per uint64_t.

#include "hamming_distance_impl.hpp"

#include <bit>

#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>

namespace hamming_distance_impl
{
void avx512(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    const simde__m256i popcount_lut
        = simde_mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const simde__m256i low_nibble_mask = simde_mm256_set1_epi8(0x0F);
    const simde__m256i zero = simde_mm256_setzero_si256();

    auto process_avx2_block = [&](std::size_t offset)
    {
        const simde__m256i vec_a = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(a_ptr + offset));
        const simde__m256i vec_b = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(b_ptr + offset));
        const simde__m256i vec_xor = simde_mm256_xor_si256(vec_a, vec_b);

        const simde__m256i low_nibbles = simde_mm256_and_si256(vec_xor, low_nibble_mask);
        const simde__m256i high_nibbles = simde_mm256_and_si256(simde_mm256_srli_epi16(vec_xor, 4), low_nibble_mask);
        const simde__m256i popcount_low = simde_mm256_shuffle_epi8(popcount_lut, low_nibbles);
        const simde__m256i popcount_high = simde_mm256_shuffle_epi8(popcount_lut, high_nibbles);
        const simde__m256i popcount_per_byte = simde_mm256_add_epi8(popcount_low, popcount_high);
        const simde__m256i lane_counts = simde_mm256_sad_epu8(popcount_per_byte, zero);

        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i *>(results_ptr + offset), lane_counts);
    };

    std::size_t i = 0;
    for (; i + 7 < size; i += 8)
    {
        process_avx2_block(i);
        process_avx2_block(i + 4);
    }

#if defined(__clang__)
#    pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}
} // namespace hamming_distance_impl
