#include <benchmark/benchmark.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "hamming_distance.hpp"

namespace
{
constexpr std::size_t kBlockSize = 1000;

void BM_HammingDistance(benchmark::State & state)
{
    std::vector<std::uint64_t> lhs_values(kBlockSize);
    std::vector<std::uint64_t> rhs_values(kBlockSize);
    std::vector<std::uint64_t> results(kBlockSize);

    std::mt19937_64 rng{std::random_device{}()};
    for (std::size_t i = 0; i < kBlockSize; ++i)
    {
        lhs_values[i] = rng();
        rhs_values[i] = rng();
    }

    double average_hamming_distance = 0.0;
    for (auto _ : state)
    {
        hamming_distance(lhs_values, rhs_values, results);

        benchmark::DoNotOptimize(results.data());

        const auto total_distance = std::accumulate(results.begin(), results.end(), std::uint64_t{0});
        average_hamming_distance = static_cast<double>(total_distance) / static_cast<double>(kBlockSize);
        benchmark::DoNotOptimize(average_hamming_distance);
    }

    state.counters["avg_hamming_distance"] = average_hamming_distance;
}

void BM_HammingDistancePattern(benchmark::State & state, std::uint64_t xor_mask)
{
    std::vector<std::uint64_t> lhs_values(kBlockSize);
    std::vector<std::uint64_t> rhs_values(kBlockSize);
    std::vector<std::uint64_t> results(kBlockSize);

    std::mt19937_64 rng{std::random_device{}()};
    for (std::size_t i = 0; i < kBlockSize; ++i)
    {
        lhs_values[i] = rng();
        rhs_values[i] = lhs_values[i] ^ xor_mask;
    }

    double average_hamming_distance = 0.0;
    for (auto _ : state)
    {
        hamming_distance(lhs_values, rhs_values, results);
        benchmark::DoNotOptimize(results.data());

        std::uint64_t total_distance = 0;
        for (const auto value : results)
        {
            total_distance += value;
        }

        average_hamming_distance = static_cast<double>(total_distance) / static_cast<double>(kBlockSize);
        benchmark::DoNotOptimize(average_hamming_distance);
    }

    state.counters["avg_hamming_distance"] = average_hamming_distance;
}

void BM_HammingDistanceAllBitsSame(benchmark::State & state)
{
    BM_HammingDistancePattern(state, 0x0000000000000000ULL);
}

void BM_HammingDistanceSomeBitsDifferent(benchmark::State & state)
{
    BM_HammingDistancePattern(state, 0x00000000000000FFULL);
}

void BM_HammingDistanceHalfBitsDifferent(benchmark::State & state)
{
    BM_HammingDistancePattern(state, 0x00000000FFFFFFFFULL);
}

void BM_HammingDistance75PercentBitsDifferent(benchmark::State & state)
{
    BM_HammingDistancePattern(state, 0x0000FFFFFFFFFFFFULL);
}

void BM_HammingDistanceAllBitsDifferent(benchmark::State & state)
{
    BM_HammingDistancePattern(state, 0xFFFFFFFFFFFFFFFFULL);
}
} // namespace

BENCHMARK(BM_HammingDistance);
BENCHMARK(BM_HammingDistanceAllBitsSame);
BENCHMARK(BM_HammingDistanceSomeBitsDifferent);
BENCHMARK(BM_HammingDistanceHalfBitsDifferent);
BENCHMARK(BM_HammingDistance75PercentBitsDifferent);
BENCHMARK(BM_HammingDistanceAllBitsDifferent);
BENCHMARK_MAIN();
