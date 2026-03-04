// SIMD popcount strategy used by the SSSE3/AVX/AVX2/AVX512 implementations below.
// For each vector chunk, compute `x = a ^ b` so set bits in `x` are differing bits.
// Each byte in `x` is split into low/high 4-bit nibbles, and each nibble (0..15)
// is mapped through a 16-entry popcount LUT via byte-wise shuffle instructions.
// Adding low+high lookup results gives per-byte bit counts (0..8); `sad_epu8` then
// sums those byte counts within each 64-bit lane, yielding one Hamming distance per uint64_t.

#include "hamming_distance.hpp"

#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>

#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse4.2.h>
#include <simde/x86/ssse3.h>

namespace
{
using HammingDistanceImpl = void (*)(const std::uint64_t *, const std::uint64_t *, std::uint64_t *, std::size_t);

void hamming_distance_scalar(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_auto_vectorized(
    const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
#if defined(__clang__)
#    pragma clang loop vectorize(enable)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_sse2(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    const simde__m128i m1 = simde_mm_set1_epi8(0x55);
    const simde__m128i m2 = simde_mm_set1_epi8(0x33);
    const simde__m128i m4 = simde_mm_set1_epi8(0x0F);
    const simde__m128i zero = simde_mm_setzero_si128();

    std::size_t i = 0;
    for (; i + 1 < size; i += 2)
    {
        const simde__m128i vec_a = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i *>(a_ptr + i));
        const simde__m128i vec_b = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i *>(b_ptr + i));
        const simde__m128i vec_xor = simde_mm_xor_si128(vec_a, vec_b);

        simde__m128i byte_counts = simde_mm_sub_epi8(vec_xor, simde_mm_and_si128(simde_mm_srli_epi16(vec_xor, 1), m1));
        byte_counts = simde_mm_add_epi8(simde_mm_and_si128(byte_counts, m2), simde_mm_and_si128(simde_mm_srli_epi16(byte_counts, 2), m2));
        byte_counts = simde_mm_and_si128(simde_mm_add_epi8(byte_counts, simde_mm_srli_epi16(byte_counts, 4)), m4);

        const simde__m128i lane_counts = simde_mm_sad_epu8(byte_counts, zero);
        simde_mm_storeu_si128(reinterpret_cast<simde__m128i *>(results_ptr + i), lane_counts);
    }

#if defined(__clang__)
#    pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_ssse3(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    const simde__m128i popcount_lut = simde_mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const simde__m128i low_nibble_mask = simde_mm_set1_epi8(0x0F);
    const simde__m128i zero = simde_mm_setzero_si128();

    std::size_t i = 0;
    for (; i + 1 < size; i += 2)
    {
        const simde__m128i vec_a = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i *>(a_ptr + i));
        const simde__m128i vec_b = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i *>(b_ptr + i));
        const simde__m128i vec_xor = simde_mm_xor_si128(vec_a, vec_b);

        const simde__m128i low_nibbles = simde_mm_and_si128(vec_xor, low_nibble_mask);
        const simde__m128i high_nibbles = simde_mm_and_si128(simde_mm_srli_epi16(vec_xor, 4), low_nibble_mask);
        const simde__m128i popcount_low = simde_mm_shuffle_epi8(popcount_lut, low_nibbles);
        const simde__m128i popcount_high = simde_mm_shuffle_epi8(popcount_lut, high_nibbles);
        const simde__m128i popcount_per_byte = simde_mm_add_epi8(popcount_low, popcount_high);
        const simde__m128i lane_counts = simde_mm_sad_epu8(popcount_per_byte, zero);

        alignas(16) std::uint64_t lanes[2];
        simde_mm_storeu_si128(reinterpret_cast<simde__m128i *>(lanes), lane_counts);

        results_ptr[i] = lanes[0];
        results_ptr[i + 1] = lanes[1];
    }

#if defined(__clang__)
#    pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_sse42(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    hamming_distance_ssse3(a_ptr, b_ptr, results_ptr, size);
}

void hamming_distance_avx(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    const simde__m256i popcount_lut
        = simde_mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const simde__m256i low_nibble_mask = simde_mm256_set1_epi8(0x0F);
    const simde__m256i zero = simde_mm256_setzero_si256();

    std::size_t i = 0;
    for (; i + 3 < size; i += 4)
    {
        const simde__m256i vec_a = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(a_ptr + i));
        const simde__m256i vec_b = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(b_ptr + i));
        const simde__m256i vec_xor = simde_mm256_xor_si256(vec_a, vec_b);

        const simde__m256i low_nibbles = simde_mm256_and_si256(vec_xor, low_nibble_mask);
        const simde__m256i high_nibbles = simde_mm256_and_si256(simde_mm256_srli_epi16(vec_xor, 4), low_nibble_mask);
        const simde__m256i popcount_low = simde_mm256_shuffle_epi8(popcount_lut, low_nibbles);
        const simde__m256i popcount_high = simde_mm256_shuffle_epi8(popcount_lut, high_nibbles);
        const simde__m256i popcount_per_byte = simde_mm256_add_epi8(popcount_low, popcount_high);
        const simde__m256i lane_counts = simde_mm256_sad_epu8(popcount_per_byte, zero);

        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i *>(results_ptr + i), lane_counts);
    }

#if defined(__clang__)
#    pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_avx2(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
{
    const simde__m256i popcount_lut
        = simde_mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const simde__m256i low_nibble_mask = simde_mm256_set1_epi8(0x0F);
    const simde__m256i zero = simde_mm256_setzero_si256();

    std::size_t i = 0;
    for (; i + 3 < size; i += 4)
    {
        const simde__m256i vec_a = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(a_ptr + i));
        const simde__m256i vec_b = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i *>(b_ptr + i));
        const simde__m256i vec_xor = simde_mm256_xor_si256(vec_a, vec_b);

        const simde__m256i low_nibbles = simde_mm256_and_si256(vec_xor, low_nibble_mask);
        const simde__m256i high_nibbles = simde_mm256_and_si256(simde_mm256_srli_epi16(vec_xor, 4), low_nibble_mask);
        const simde__m256i popcount_low = simde_mm256_shuffle_epi8(popcount_lut, low_nibbles);
        const simde__m256i popcount_high = simde_mm256_shuffle_epi8(popcount_lut, high_nibbles);
        const simde__m256i popcount_per_byte = simde_mm256_add_epi8(popcount_low, popcount_high);
        const simde__m256i lane_counts = simde_mm256_sad_epu8(popcount_per_byte, zero);

        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i *>(results_ptr + i), lane_counts);
    }

#if defined(__clang__)
#    pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i)
    {
        results_ptr[i] = static_cast<std::uint64_t>(std::popcount(a_ptr[i] ^ b_ptr[i]));
    }
}

void hamming_distance_avx512(const std::uint64_t * a_ptr, const std::uint64_t * b_ptr, std::uint64_t * results_ptr, std::size_t size)
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

bool parse_truthy_env_value(const char * value)
{
    if (value == nullptr)
    {
        return false;
    }

    std::string lowered;
    for (const unsigned char c : std::string{value})
    {
        lowered.push_back(static_cast<char>(std::tolower(c)));
    }

    return lowered == "1" || lowered == "true" || lowered == "on" || lowered == "yes";
}

HammingDistanceImpl select_runtime_dispatch_impl()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512f"))
    {
        return &hamming_distance_avx512;
    }

    if (__builtin_cpu_supports("avx2"))
    {
        return &hamming_distance_avx2;
    }

    if (__builtin_cpu_supports("avx"))
    {
        return &hamming_distance_avx;
    }

    if (__builtin_cpu_supports("sse4.2"))
    {
        return &hamming_distance_sse42;
    }

    if (__builtin_cpu_supports("ssse3"))
    {
        return &hamming_distance_ssse3;
    }

    if (__builtin_cpu_supports("sse2"))
    {
        return &hamming_distance_sse2;
    }
#    endif
#endif

    return &hamming_distance_scalar;
}

HammingDistanceImpl select_active_impl()
{
    const bool prefer_auto_vectorization = parse_truthy_env_value(std::getenv("HAMMING_DISTANCE_PREFER_AUTO_VECTORIZATION"));

    if (prefer_auto_vectorization)
    {
        return &hamming_distance_auto_vectorized;
    }

    return select_runtime_dispatch_impl();
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
