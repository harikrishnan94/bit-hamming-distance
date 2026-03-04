#include "hamming_distance.hpp"

#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>

#include <simde/x86/avx2.h>

void hamming_distance(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b, std::span<std::uint64_t> results)
{
    if (a.size() != b.size() || a.size() != results.size())
    {
        throw std::invalid_argument{"Input spans must have the same size."};
    }

    std::size_t i = 0;
    const auto size = b.size();
    const auto * a_ptr = a.data();
    const auto * b_ptr = b.data();
    auto * results_ptr = results.data();

#if PREFER_AUTO_VECTORIZATION == 0
    // 4-bit popcount lookup table indexed via byte-wise shuffle.
    const simde__m256i popcount_lut
        = simde_mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const simde__m256i low_nibble_mask = simde_mm256_set1_epi8(0x0F);
    const simde__m256i zero = simde_mm256_setzero_si256();

    auto process_vector_pair
        = [&](const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t offset)
    {
        // First 256-bit chunk (4x uint64_t): load and XOR.
        const auto vec_a = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(a_ptr + offset));
        const auto vec_b = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(b_ptr + offset));
        const auto vec_x = simde_mm256_xor_si256(vec_a, vec_b);

        // Split bytes into low/high nibbles (0..15), lookup popcount, and accumulate per byte.
        const auto low_nibbles = simde_mm256_and_si256(vec_x, low_nibble_mask);
        const auto high_nibbles = simde_mm256_and_si256(simde_mm256_srli_epi16(vec_x, 4), low_nibble_mask);
        const auto popcount_low = simde_mm256_shuffle_epi8(popcount_lut, low_nibbles);
        const auto popcount_high = simde_mm256_shuffle_epi8(popcount_lut, high_nibbles);
        const auto popcount_per_byte = simde_mm256_add_epi8(popcount_low, popcount_high);

        // Horizontal byte sums within each 64-bit lane (4 results).
        const auto popcount_vec = simde_mm256_sad_epu8(popcount_per_byte, zero);
        // Store results
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i *>(results_ptr + offset), popcount_vec);
    };

    for (; i + 7 < size; i += 8)
    {
        process_vector_pair(a_ptr, b_ptr, results_ptr, i);
        process_vector_pair(a_ptr, b_ptr, results_ptr, i + 4);
    }

    for (; i + 3 < size; i += 4)
    {
        process_vector_pair(a_ptr, b_ptr, results_ptr, i);
        process_vector_pair(a_ptr, b_ptr, results_ptr, i + 4);
    }

#    pragma clang loop vectorize(disable)
#endif

    for (; i < size; ++i)
    {
        // Scalar tail for remaining elements.
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}
