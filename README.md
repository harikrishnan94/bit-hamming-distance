# bit_hamming_distance

A high-performance C++23 implementation of 64-bit Hamming distance over arrays, with selectable SIMD implementations and benchmark targets.

## Features

- Computes per-element Hamming distance for two `uint64_t` spans.
- Explicit implementation selection via environment variable.
- Includes:
  - Google Benchmark target (`benchmark_bit_hamming_distance`)
  - Lightweight timing benchmark (`simple_bench_bit_hamming_distance`)
  - Assembly generation target (`hamming_distance_asm`)

## Requirements

- CMake `>= 3.20`
- C++23-compatible compiler (Clang or GCC recommended)
- Internet access for first configure (FetchContent pulls dependencies):
  - `google/benchmark` (`v1.9.1`)
  - `simd-everywhere/simde` (`v0.8.2`)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run Benchmarks

Google Benchmark:

```bash
./build/benchmark_bit_hamming_distance
```

Simple timing benchmark:

```bash
./build/simple_bench_bit_hamming_distance
```

## API

Header: `include/hamming_distance.hpp`

```cpp
void hamming_distance(
    std::span<const std::uint64_t> a,
    std::span<const std::uint64_t> b,
    std::span<std::uint64_t> results);
```

### Behavior

- `a.size()`, `b.size()`, and `results.size()` must match.
- Throws `std::invalid_argument` if sizes differ.
- Writes per-element bit distance into `results`.

### Example

```cpp
#include <cstdint>
#include <span>
#include <vector>

#include "hamming_distance.hpp"

int main() {
    std::vector<std::uint64_t> a{0b1011, 0xFFFFFFFFFFFFFFFFULL};
    std::vector<std::uint64_t> b{0b0011, 0x0ULL};
    std::vector<std::uint64_t> out(a.size());

    hamming_distance(a, b, out);
    // out[0] == 1, out[1] == 64
}
```

## Implementation Selection

By default, the auto-vectorized implementation is used.

To explicitly select an implementation:

```bash
HAMMING_DISTANCE_IMPLEMENTATION=avx2 ./build/simple_bench_bit_hamming_distance
```

Supported values (case-insensitive, separators ignored): `scalar`, `auto`, `avx2`, `avx512`.

## License

MIT. See [LICENSE](LICENSE).
