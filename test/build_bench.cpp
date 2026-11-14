// Benchmark for BVH construction: measures build time and tree quality (SAH cost)
// on procedurally generated scenes of 100k / 500k / 1M triangles.

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/bbox.h>
#include <bvh/v2/node.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

using Scalar = float;
using Vec3   = bvh::v2::Vec<Scalar, 3>;
using BBox   = bvh::v2::BBox<Scalar, 3>;
using Tri    = bvh::v2::Tri<Scalar, 3>;
using Node   = bvh::v2::Node<Scalar, 3>;
using Bvh    = bvh::v2::Bvh<Node>;

// Deterministic PRNG (xorshift64*)
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed * 2685821657736338717ull + 1) {}
    uint64_t next_u64() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ull;
    }
    Scalar next_f01() { return static_cast<Scalar>((next_u64() >> 40) & 0xFFFFFF) / Scalar(0x1000000); }
    Scalar next_f(Scalar lo, Scalar hi) { return lo + (hi - lo) * next_f01(); }
};

// Generates a scene that mimics the spatial distribution of real models:
// many small, dense clusters of triangles plus a sparse background.
static std::vector<Tri> generate_scene(size_t tri_count, uint64_t seed) {
    Rng rng(seed);
    std::vector<Tri> tris;
    tris.reserve(tri_count);

    const size_t cluster_count = std::max<size_t>(1, tri_count / 2000);
    const size_t background_count = tri_count / 20;

    for (size_t c = 0; c < cluster_count; ++c) {
        Vec3 center(rng.next_f(-10, 10), rng.next_f(-10, 10), rng.next_f(-10, 10));
        Scalar radius = rng.next_f(Scalar(0.05), Scalar(1.5));
        size_t count = (c + 1 == cluster_count)
            ? tri_count - background_count - tris.size()
            : tri_count / cluster_count - background_count / cluster_count;
        for (size_t i = 0; i < count; ++i) {
            Vec3 p(
                center[0] + rng.next_f(-radius, radius),
                center[1] + rng.next_f(-radius, radius),
                center[2] + rng.next_f(-radius, radius));
            Scalar s = radius * Scalar(0.02);
            Vec3 u(rng.next_f(-s, s), rng.next_f(-s, s), rng.next_f(-s, s));
            Vec3 v(rng.next_f(-s, s), rng.next_f(-s, s), rng.next_f(-s, s));
            tris.emplace_back(p, p + u, p + v);
        }
    }
    for (size_t i = 0; i < background_count; ++i) {
        Vec3 p(rng.next_f(-10, 10), rng.next_f(-10, 10), rng.next_f(-10, 10));
        Scalar s = Scalar(0.05);
        Vec3 u(rng.next_f(-s, s), rng.next_f(-s, s), rng.next_f(-s, s));
        Vec3 v(rng.next_f(-s, s), rng.next_f(-s, s), rng.next_f(-s, s));
        tris.emplace_back(p, p + u, p + v);
    }
    return tris;
}

// Standard SAH cost: traversal cost for inner nodes + intersection cost for leaves,
// normalized by the root area (cost_ratio = 1, matching the library default).
static Scalar compute_sah_cost(const Bvh& bvh) {
    Scalar inner_cost = 0, leaf_cost = 0;
    for (const auto& node : bvh.nodes) {
        auto area = node.get_bbox().get_half_area();
        if (node.is_leaf())
            leaf_cost += area * static_cast<Scalar>(node.index.prim_count());
        else
            inner_cost += area;
    }
    return (inner_cost + leaf_cost) / bvh.get_root().get_bbox().get_half_area();
}

template <typename F>
static double time_ms(F&& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    size_t iters = 3;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc)
            iters = std::strtoul(argv[++i], nullptr, 10);
    }

    const size_t scales[] = { 100000, 500000, 1000000 };
    std::printf("%10s | %12s | %12s | %10s\n", "triangles", "build (ms)", "SAH cost", "nodes");
    std::printf("-----------+--------------+--------------+----------\n");

    bvh::v2::ThreadPool thread_pool;
    for (size_t tri_count : scales) {
        auto tris = generate_scene(tri_count, 42);

        std::vector<BBox> bboxes(tris.size());
        std::vector<Vec3> centers(tris.size());
        bvh::v2::ParallelExecutor executor(thread_pool);
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i]  = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;

        Bvh bvh;
        std::vector<double> times;
        for (size_t i = 0; i < iters; ++i) {
            times.push_back(time_ms([&] {
                bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);
            }));
        }
        std::sort(times.begin(), times.end());
        double median = times[times.size() / 2];

        std::printf("%10zu | %12.1f | %12.4f | %10zu\n",
            tri_count, median, static_cast<double>(compute_sah_cost(bvh)), bvh.nodes.size());
        std::fflush(stdout);
    }
    return 0;
}
