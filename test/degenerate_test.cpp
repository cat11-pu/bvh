#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/validate.h>

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;
using Quality = bvh::v2::DefaultBuilder<Node>::Quality;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
static constexpr size_t stack_size = 64;

// Always-on check (asserts are compiled out under NDEBUG, which would leave the helpers unused).
#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "Check failed: " #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

struct Stats {
    size_t build_count = 0;
    size_t total_input = 0;
    size_t total_degenerate = 0;
    size_t total_valid = 0;
    size_t ray_count = 0;
    size_t mismatch_count = 0;
};

static Stats g_stats;

// Generates the different kinds of degenerate triangles requested in the task.
static Tri make_degenerate(std::mt19937& rng, int kind) {
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    Vec3 a(dist(rng), dist(rng), dist(rng));
    auto distinct_point = [&] {
        Vec3 b;
        do {
            b = Vec3(dist(rng), dist(rng), dist(rng));
        } while (dot(b - a, b - a) == Scalar(0));
        return b;
    };
    switch (kind % 4) {
        case 0: return Tri(a, a, a);                       // All vertices identical (point)
        case 1: return Tri(a, a, distinct_point());         // Two identical vertices
        case 2: return Tri(a, distinct_point(), a);         // Repeated first/last vertex
        default: {                                         // Collinear, distinct vertices
            // Exact integer arithmetic keeps the three points guaranteed collinear.
            Vec3 b = Vec3(a[0] + 1, a[1], a[2]);
            Vec3 c = Vec3(a[0] + 2, a[1], a[2]);
            return Tri(a, b, c);
        }
    }
    return Tri(a, a, a);
}

static Tri make_valid(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    for (int attempt = 0; attempt < 16; ++attempt) {
        Tri tri(
            Vec3(dist(rng), dist(rng), dist(rng)),
            Vec3(dist(rng), dist(rng), dist(rng)),
            Vec3(dist(rng), dist(rng), dist(rng)));
        if (!bvh::v2::is_degenerate(tri))
            return tri;
    }
    // Guaranteed non-degenerate fallback triangle.
    return Tri(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0));
}

struct Scene {
    std::vector<Tri> tris;
    size_t expected_degenerate = 0;
};

static Scene make_scene(size_t valid_count, size_t degenerate_count, unsigned seed) {
    std::mt19937 rng(seed);
    Scene scene;
    scene.tris.reserve(valid_count + degenerate_count);
    for (size_t i = 0; i < valid_count; ++i)
        scene.tris.push_back(make_valid(rng));
    for (size_t i = 0; i < degenerate_count; ++i) {
        auto tri = make_degenerate(rng, static_cast<int>(i));
        CHECK(bvh::v2::is_degenerate(tri));
        scene.tris.push_back(tri);
        scene.expected_degenerate++;
    }
    // Shuffle so degenerate and valid primitives are interleaved.
    std::shuffle(scene.tris.begin(), scene.tris.end(), rng);
    // Recompute the expected count from the actual data to be robust against RNG collisions.
    scene.expected_degenerate = 0;
    for (const auto& tri : scene.tris)
        if (bvh::v2::is_degenerate(tri))
            scene.expected_degenerate++;
    return scene;
}

// Brute-force closest-hit query over every (valid) primitive, no acceleration structure.
static size_t brute_force_intersect(
    const std::vector<PrecomputedTri>& valid_tris,
    Ray ray)
{
    size_t hit_id = invalid_id;
    for (size_t i = 0; i < valid_tris.size(); ++i) {
        if (auto hit = valid_tris[i].intersect(ray)) {
            ray.tmax = std::get<0>(*hit);
            hit_id = i;
        }
    }
    return hit_id;
}

static size_t bvh_intersect(const Bvh& bvh, const std::vector<PrecomputedTri>& valid_tris, Ray ray) {
    size_t hit_id = invalid_id;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    bvh.intersect<false, false>(ray, bvh.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                if (auto hit = valid_tris[i].intersect(ray)) {
                    ray.tmax = std::get<0>(*hit);
                    hit_id = i;
                }
            }
            return hit_id != invalid_id;
        });
    return hit_id;
}

static bool check_invariants(
    const Bvh& bvh,
    const bvh::v2::ValidatedScene<Node>& scene,
    const Scene& input,
    size_t reported_degenerate)
{
    bool ok = true;
    ok &= reported_degenerate == input.expected_degenerate;
    ok &= scene.degenerate_count == input.expected_degenerate;
    ok &= scene.bboxes.size() == input.tris.size() - input.expected_degenerate;
    ok &= scene.centers.size() == scene.bboxes.size();
    ok &= scene.valid_ids.size() == scene.bboxes.size();
    ok &= bvh.prim_ids.size() == scene.bboxes.size();

    // An all-degenerate (or empty) input produces an empty BVH; nothing more to check.
    if (bvh.prim_ids.empty()) {
        return ok && bvh.nodes.empty();
    }

    // Every primitive referenced by the BVH must be valid (non-degenerate).
    for (size_t id : bvh.prim_ids) {
        ok &= id < scene.valid_ids.size();
        if (id < scene.valid_ids.size())
            ok &= !bvh::v2::is_degenerate(input.tris[scene.valid_ids[id]]);
    }

    // The set of referenced primitives must be exactly the valid primitives, each once.
    std::vector<bool> seen(scene.valid_ids.size(), false);
    for (size_t id : bvh.prim_ids) {
        ok &= !seen[id];
        seen[id] = true;
    }
    for (bool entry_seen : seen)
        ok &= entry_seen;
    return ok;
}

static void cross_check_traversal(
    const Bvh& bvh,
    const std::vector<PrecomputedTri>& valid_tris,
    unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> origin(-5.f, 5.f);
    std::normal_distribution<float> normal_dir(0.f, 1.f);
    constexpr size_t rays_per_case = 2000;

    for (size_t r = 0; r < rays_per_case; ++r) {
        Ray ray(
            Vec3(origin(rng), origin(rng), origin(rng)),
            Vec3(normal_dir(rng), normal_dir(rng), normal_dir(rng)),
            0.f,
            std::numeric_limits<Scalar>::max());

        auto expected = brute_force_intersect(valid_tris, ray);
        auto actual   = bvh_intersect(bvh, valid_tris, ray);
        g_stats.ray_count++;
        if (expected != actual) {
            g_stats.mismatch_count++;
            std::cerr << "Mismatch on ray " << r << ": brute=" << expected
                      << " bvh=" << actual << std::endl;
        }
        CHECK(expected == actual);
    }
}

static void run_case(
    const std::string& name,
    const Scene& input,
    Quality quality,
    bool parallel,
    unsigned seed)
{
    bvh::v2::ThreadPool thread_pool;
    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = quality;
    config.parallel_threshold = parallel ? 1 : 1024;

    auto scene = bvh::v2::validate_triangles<Node>(
        std::span<const Tri>(input.tris.data(), input.tris.size()));

    size_t reported = invalid_id;
    Bvh bvh = parallel
        ? bvh::v2::build_validated<Node>(thread_pool,
            std::span<const Tri>(input.tris.data(), input.tris.size()), config, &reported)
        : bvh::v2::build_validated<Node>(
            std::span<const Tri>(input.tris.data(), input.tris.size()), config, &reported);

    CHECK(check_invariants(bvh, scene, input, reported));

    // Build the precomputed triangles in the same permutation as the BVH leaves.
    std::vector<PrecomputedTri> valid_tris;
    valid_tris.reserve(bvh.prim_ids.size());
    for (size_t local_id : bvh.prim_ids)
        valid_tris.emplace_back(input.tris[scene.valid_ids[local_id]]);

    if (!valid_tris.empty())
        cross_check_traversal(bvh, valid_tris, seed);

    g_stats.build_count++;
    g_stats.total_input += input.tris.size();
    g_stats.total_degenerate += reported;
    g_stats.total_valid += bvh.prim_ids.size();

    std::cout
        << "[ok] " << name
        << " quality=" << static_cast<int>(quality)
        << " parallel=" << (parallel ? 1 : 0)
        << " input=" << input.tris.size()
        << " skipped=" << reported
        << " built=" << bvh.prim_ids.size()
        << " nodes=" << bvh.nodes.size()
        << std::endl;
}

int main() {
    const Quality qualities[] = { Quality::Low, Quality::Medium, Quality::High };

    // 1. Empty scene: must build an (empty) BVH and traverse without crashing.
    {
        Scene empty = make_scene(0, 0, 1);
        for (auto quality : qualities)
            for (bool parallel : { false, true })
                run_case("empty", empty, quality, parallel, 100);

        // Explicitly exercise traversal of the empty BVH.
        Bvh empty_bvh;
        bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
        Ray ray(Vec3(0), Vec3(0, 0, 1));
        empty_bvh.intersect<false, false>(ray, Bvh::Index(0), stack,
            [&] (size_t, size_t) { return false; });
        std::cout << "[ok] empty BVH traversal" << std::endl;
    }

    // 2. Massive fully-degenerate scenes (all four degenerate shapes), no valid primitive.
    for (auto quality : qualities)
        for (bool parallel : { false, true })
            run_case("all-degenerate", make_scene(0, 10000, 2), quality, parallel, 101);

    // 3. Mixed scenes with a large majority of degenerate primitives.
    for (auto quality : qualities)
        for (bool parallel : { false, true })
            run_case("mostly-degenerate", make_scene(50, 9950, 3), quality, parallel, 102);

    // 4. Mixed scenes with a small number of degenerates scattered among valid ones.
    for (auto quality : qualities)
        for (bool parallel : { false, true })
            run_case("few-degenerate", make_scene(5000, 64, 4), quality, parallel, 103);

    // 5. Valid-only scene (degenerate count must be zero, regression check).
    for (auto quality : qualities)
        for (bool parallel : { false, true })
            run_case("valid-only", make_scene(2000, 0, 5), quality, parallel, 104);

    // 6. A single degenerate and a single valid primitive.
    for (auto quality : qualities)
        for (bool parallel : { false, true }) {
            run_case("single-degenerate", make_scene(0, 1, 6), quality, parallel, 105);
            run_case("single-valid", make_scene(1, 0, 7), quality, parallel, 106);
        }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "builds:            " << g_stats.build_count << std::endl;
    std::cout << "input triangles:   " << g_stats.total_input << std::endl;
    std::cout << "skipped (degen.):  " << g_stats.total_degenerate << std::endl;
    std::cout << "built primitives:  " << g_stats.total_valid << std::endl;
    std::cout << "rays cross-checked:" << g_stats.ray_count << std::endl;
    std::cout << "mismatches:        " << g_stats.mismatch_count << std::endl;

    CHECK(g_stats.mismatch_count == 0);
    std::cout << "All degenerate-triangle tests passed." << std::endl;
    return 0;
}
