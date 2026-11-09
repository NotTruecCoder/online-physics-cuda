#pragma once

#include <omp.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

// ONE LATTICE, ALL CORES, SAME BITS. Every elementwise op and every stencil computes a cell from the
// pre-state alone, so cells can be computed in any order by any thread and come out identical. The
// lattice is cut into FIXED blocks (never "one block per thread"), so even a reduction — summed per
// block, then over the blocks in order — gives the same bits on 1 core or 64.
namespace fieldengine::field {

inline constexpr std::size_t kParallelBlock = 8192;

// WHAT FADES REACHES ZERO. A field that decays exponentially passes through denormal floats on its way down,
// which CPUs compute with far more slowly. A precaution for long sessions: scent at its 0.06/s fade gets there
// after some 24 minutes; over the first minute they are too rare to measure. Every thread that computes cells
// flushes them, so the bits stay the same on any thread count.
inline void flush_denormals()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// A few threads, not all of them: this shares the machine with the renderer (and whatever else runs
// there), and a barrier waiting on one descheduled thread costs more than the extra threads win.
// OMP_NUM_THREADS overrides. The bits are the same whatever the count.
inline void limit_threads()
{
    // Per thread: the count is a property of the thread that starts a parallel region.
    thread_local const bool done = [] {
        flush_denormals();
        if (std::getenv("OMP_NUM_THREADS") == nullptr) {
            omp_set_num_threads(std::clamp(omp_get_num_procs() / 2, 1, 4));
        }
        return true;
    }();
    (void)done;
}

// fn(begin, end) over [0, n) in fixed blocks.
template <typename Fn>
void parallel_blocks(std::size_t n, Fn&& fn)
{
    const long long blocks = static_cast<long long>((n + kParallelBlock - 1) / kParallelBlock);
    if (blocks <= 1) {
        fn(std::size_t{0}, n);
        return;
    }
#pragma omp parallel for schedule(static)
    for (long long b = 0; b < blocks; ++b) {
        flush_denormals();
        const std::size_t begin = static_cast<std::size_t>(b) * kParallelBlock;
        fn(begin, std::min(n, begin + kParallelBlock));
    }
}

// fn(i, x, y, z) for every cell of an nx*ny*nz lattice (x fastest), one z-plane per task.
template <typename Fn>
void parallel_cells(int nx, int ny, int nz, Fn&& fn)
{
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    auto run_plane = [&](int z) {
        flush_denormals();
        std::size_t i = static_cast<std::size_t>(z) * plane;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x, ++i) {
                fn(i, x, y, z);
            }
        }
    };
    if (nz < 2 || plane * static_cast<std::size_t>(nz) < kParallelBlock) {
        for (int z = 0; z < nz; ++z) {
            run_plane(z);
        }
        return;
    }
#pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; ++z) {
        run_plane(z);
    }
}

// Sum of fn(i) over [0, n): per fixed block, then the blocks in order.
template <typename Fn>
double parallel_sum(std::size_t n, Fn&& fn)
{
    const std::size_t blocks = (n + kParallelBlock - 1) / kParallelBlock;
    std::vector<double> partial(blocks, 0.0);
    parallel_blocks(n, [&](std::size_t begin, std::size_t end) {
        double acc = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            acc += fn(i);
        }
        partial[begin / kParallelBlock] = acc;
    });
    double sum = 0.0;
    for (const double p : partial) {
        sum += p;
    }
    return sum;
}

} // namespace fieldengine::field
