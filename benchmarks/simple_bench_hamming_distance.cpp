#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "hamming_distance.hpp"


namespace
{
constexpr std::size_t kBlockSize = 1000;
constexpr std::uint64_t kWarmupIters = 20'000;
constexpr std::uint64_t kMeasureIters = 200'000;

struct BlockData
{
    std::vector<std::uint64_t> lhs;
    std::vector<std::uint64_t> rhs;
    std::vector<std::uint64_t> results;
};

BlockData make_random_block()
{
    BlockData data{std::vector<std::uint64_t>(kBlockSize), std::vector<std::uint64_t>(kBlockSize), std::vector<std::uint64_t>(kBlockSize)};

    std::mt19937_64 rng{0x1234'5678'9ABC'DEF0ULL};
    for (std::size_t i = 0; i < kBlockSize; ++i)
    {
        data.lhs[i] = rng();
        data.rhs[i] = rng();
    }
    return data;
}

BlockData make_pattern_block(std::uint64_t xor_mask)
{
    BlockData data{std::vector<std::uint64_t>(kBlockSize), std::vector<std::uint64_t>(kBlockSize), std::vector<std::uint64_t>(kBlockSize)};

    std::mt19937_64 rng{0x0FED'CBA9'8765'4321ULL};
    for (std::size_t i = 0; i < kBlockSize; ++i)
    {
        data.lhs[i] = rng();
        data.rhs[i] = data.lhs[i] ^ xor_mask;
    }
    return data;
}

double measure_average_ns_per_block(BlockData & data)
{
    volatile std::uint64_t sink = 0;

    for (std::uint64_t i = 0; i < kWarmupIters; ++i)
    {
        data.lhs[0] ^= (i + 1U);
        hamming_distance(data.lhs, data.rhs, data.results);
        sink ^= data.results[0];
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < kMeasureIters; ++i)
    {
        data.lhs[0] ^= (i + 0x9E37'79B9U);
        hamming_distance(data.lhs, data.rhs, data.results);
        sink ^= data.results[0];
    }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (sink == 0xFFFF'FFFF'FFFF'FFFFULL)
    {
        std::cerr << "ignore: " << sink << '\n';
    }

    return static_cast<double>(total_ns) / static_cast<double>(kMeasureIters);
}

void print_line(std::string_view name, double avg_ns)
{
    std::cout << std::format("{:36} {:10.2f} ns\n", name, avg_ns);
}
} // namespace

int main()
{
    auto random_block = make_random_block();
    auto all_bits_same = make_pattern_block(0x0000000000000000ULL);
    auto some_bits_different = make_pattern_block(0x00000000000000FFULL);
    auto half_bits_different = make_pattern_block(0x00000000FFFFFFFFULL);
    auto seventy_five_percent_bits_different = make_pattern_block(0x0000FFFFFFFFFFFFULL);
    auto all_bits_different = make_pattern_block(0xFFFFFFFFFFFFFFFFULL);

    print_line("random", measure_average_ns_per_block(random_block));
    print_line("all_bits_same", measure_average_ns_per_block(all_bits_same));
    print_line("some_bits_different", measure_average_ns_per_block(some_bits_different));
    print_line("half_bits_different", measure_average_ns_per_block(half_bits_different));
    print_line("75_percent_bits_different", measure_average_ns_per_block(seventy_five_percent_bits_different));
    print_line("all_bits_different", measure_average_ns_per_block(all_bits_different));

    return 0;
}
