#pragma once

#include <cstdint>
#include <span>


void hamming_distance(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b, std::span<std::uint64_t> results);