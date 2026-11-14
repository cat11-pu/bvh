#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/node.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using bvh::v2::BBox;
using bvh::v2::DefaultBuilder;
using bvh::v2::GrowingStack;
using bvh::v2::Node;
using bvh::v2::PrecomputedTri;
using bvh::v2::Ray;
using bvh::v2::ThreadPool;
using bvh::v2::Tri;
using bvh::v2::Vec;

constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

enum class Category {
    Single,
    Normal,
    Degenerate,
    Huge,
    Mixed,
    Scale
};

const char* category_name(Category category) {
    switch (category) {
        case Category::Single: return "single";
        case Category::Normal: return "normal";
        case Category::Degenerate: return "degenerate";
        case Category::Huge: return "huge";
        case Category::Mixed: return "mixed";
        case Category::Scale: return "scale";
    }
    return "unknown";
}

struct Rng {
    using result_type = uint64_t;

    uint64_t state;

    explicit Rng(uint64_t seed) : state(seed) {}

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    result_type operator () () { return next(); }

    uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        auto z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    uint64_t below(uint64_t bound) {
        return bound == 0 ? 0 : next() % bound;
    }

    double unit() {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }
};

template <typename T>
struct ScalarParams;

template <>
struct ScalarParams<float> {
    static constexpr float huge_base = 1e12f;
    static constexpr float huge_extent = 1e7f;
};

template <>
struct ScalarParams<double> {
    static constexpr double huge_base = 1e30;
    static constexpr double huge_extent = 1e24;
};

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using BvhNode = Node<T, 3>;

template <typename T>
using Bvh = bvh::v2::Bvh<BvhNode<T>>;

template <typename T>
using Triangle = Tri<T, 3>;

template <typename T>
using QueryRay = Ray<T, 3>;

template <typename T>
using TriangleBBox = BBox<T, 3>;

template <typename T>
T uniform_real(Rng& rng, T min, T max) {
    return min + (max - min) * static_cast<T>(rng.unit());
}

template <typename T>
Vec3<T> random_vec(Rng& rng, T min, T max) {
    return Vec3<T>::generate([&] (size_t) { return uniform_real(rng, min, max); });
}

template <typename T>
Vec3<T> random_unit_vec(Rng& rng) {
    for (size_t attempt = 0; attempt < 8; ++attempt) {
        auto v = random_vec<T>(rng, T(-1), T(1));
        auto len_sq = bvh::v2::dot(v, v);
        if (len_sq > std::numeric_limits<T>::min()) {
            return v * (T(1) / std::sqrt(len_sq));
        }
    }
    return Vec3<T>(T(1), T(0), T(0));
}

template <typename T>
struct Scene {
    Category category = Category::Normal;
    size_t case_id = 0;
    uint64_t seed = 0;
    std::vector<Triangle<T>> triangles;
    std::vector<QueryRay<T>> rays;
    size_t degenerate_count = 0;
    T radius = 1;
};

template <typename T>
Triangle<T> make_degenerate_triangle(
    Rng& rng,
    const Vec3<T>& center,
    T extent,
    size_t kind)
{
    auto p0 = center + random_vec<T>(rng, -extent, extent);
    auto p1 = center + random_vec<T>(rng, -extent, extent);
    auto p2 = center + random_vec<T>(rng, -extent, extent);

    switch (kind % 5) {
        case 0:
            p1 = p0;
            p2 = p0;
            break;
        case 1:
            p1 = p0;
            break;
        case 2:
            p2 = p0;
            break;
        case 3: {
            auto delta = p1 - p0;
            p2 = p0 + delta * T(2);
            break;
        }
        case 4: {
            auto midpoint = (p0 + p1) * T(0.5);
            auto offset = random_unit_vec<T>(rng) * (extent * T(1e-5));
            p2 = midpoint + offset;
            break;
        }
    }
    return Triangle<T>(p0, p1, p2);
}

template <typename T>
Triangle<T> make_regular_triangle(Rng& rng, const Vec3<T>& center, T extent) {
    auto p0 = center + random_vec<T>(rng, -extent, extent);
    auto p1 = center + random_vec<T>(rng, -extent, extent);
    auto p2 = center + random_vec<T>(rng, -extent, extent);
    return Triangle<T>(p0, p1, p2);
}

template <typename T>
Scene<T> make_scene(Category category, size_t primitive_count, Rng& rng, size_t case_id) {
    Scene<T> scene;
    scene.category = category;
    scene.case_id = case_id;
    scene.seed = rng.next();

    Vec3<T> center(T(0));
    T extent = T(1);
    T degenerate_probability = T(0);

    switch (category) {
        case Category::Single:
        case Category::Normal:
            extent = std::pow(T(10), uniform_real<T>(rng, T(-2), T(2)));
            degenerate_probability = category == Category::Single ? T(0.35) : T(0.08);
            break;
        case Category::Degenerate:
            extent = std::pow(T(10), uniform_real<T>(rng, T(-2), T(2)));
            degenerate_probability = T(0.80);
            break;
        case Category::Huge:
            for (size_t axis = 0; axis < 3; ++axis) {
                auto sign = rng.unit() < 0.5 ? T(-1) : T(1);
                center[axis] = sign * ScalarParams<T>::huge_base *
                    uniform_real<T>(rng, T(0.5), T(1));
            }
            extent = ScalarParams<T>::huge_extent * uniform_real<T>(rng, T(0.5), T(1));
            degenerate_probability = T(0.15);
            break;
        case Category::Mixed:
            extent = std::pow(T(10), uniform_real<T>(rng, T(-2), T(2)));
            degenerate_probability = T(0.45);
            break;
        case Category::Scale:
            extent = T(1);
            degenerate_probability = T(0.0625);
            break;
    }

    scene.triangles.reserve(primitive_count);
    for (size_t i = 0; i < primitive_count; ++i) {
        if (rng.unit() < static_cast<double>(degenerate_probability)) {
            scene.triangles.push_back(
                make_degenerate_triangle(rng, center, extent, static_cast<size_t>(rng.below(5))));
            ++scene.degenerate_count;
        } else {
            scene.triangles.push_back(make_regular_triangle(rng, center, extent));
        }
    }

    auto bbox = TriangleBBox<T>::make_empty();
    for (const auto& triangle : scene.triangles)
        bbox.extend(triangle.get_bbox());

    auto diagonal = bbox.get_diagonal();
    auto radius = bvh::v2::length(diagonal);
    if (!(radius > T(0)) || !std::isfinite(radius))
        radius = extent * T(4);
    scene.radius = radius;

    return scene;
}

template <typename T>
Vec3<T> random_point_in_bbox(Rng& rng, const TriangleBBox<T>& bbox) {
    return Vec3<T>::generate([&] (size_t axis) {
        return uniform_real(rng, bbox.min[axis], bbox.max[axis]);
    });
}

template <typename T>
QueryRay<T> make_ray(const Scene<T>& scene, Rng& rng) {
    auto bbox = TriangleBBox<T>::make_empty();
    for (const auto& triangle : scene.triangles)
        bbox.extend(triangle.get_bbox());

    if (!scene.triangles.empty() && rng.unit() < 0.70) {
        const auto& triangle = scene.triangles[static_cast<size_t>(rng.below(scene.triangles.size()))];
        T u = T(1) / T(3);
        T v = T(1) / T(3);
        if (rng.unit() >= 0.60) {
            u = uniform_real<T>(rng, T(0.02), T(0.70));
            v = uniform_real<T>(rng, T(0.02), T(0.96) - u);
        }
        auto target = triangle.p0 + (triangle.p1 - triangle.p0) * u +
            (triangle.p2 - triangle.p0) * v;
        auto distance = scene.radius * uniform_real<T>(rng, T(0.4), T(3.0));
        auto origin = target + random_unit_vec<T>(rng) * distance;
        auto direction = bvh::v2::normalize(target - origin);
        auto tmax = distance * T(2) + scene.radius;
        return QueryRay<T>(origin, direction, T(0), tmax);
    }

    auto origin = random_point_in_bbox(rng, bbox);
    auto target = random_point_in_bbox(rng, bbox);
    auto delta = target - origin;
    auto len_sq = bvh::v2::dot(delta, delta);
    if (!(len_sq > std::numeric_limits<T>::min())) {
        origin = bbox.min;
        target = bbox.max;
        delta = target - origin;
        len_sq = bvh::v2::dot(delta, delta);
    }
    auto direction = len_sq > std::numeric_limits<T>::min()
        ? delta * (T(1) / std::sqrt(len_sq))
        : Vec3<T>(T(1), T(0), T(0));
    auto tmax = scene.radius * T(3);
    return QueryRay<T>(origin, direction, T(0), tmax);
}

template <typename T>
struct QueryResult {
    bool hit = false;
    size_t prim_id = invalid_id;
    T t{};
    T u{};
    T v{};
};

template <typename T>
bool exact_equal(const QueryResult<T>& lhs, const QueryResult<T>& rhs) {
    if (lhs.hit != rhs.hit)
        return false;
    if (!lhs.hit)
        return true;
    return lhs.prim_id == rhs.prim_id &&
        lhs.t == rhs.t &&
        lhs.u == rhs.u &&
        lhs.v == rhs.v;
}

template <typename T>
bool order_invariant_equal(const QueryResult<T>& lhs, const QueryResult<T>& rhs) {
    if (lhs.hit != rhs.hit)
        return false;
    return !lhs.hit || lhs.t == rhs.t;
}

template <typename T>
QueryResult<T> intersect_bruteforce(
    std::span<const Triangle<T>> triangles,
    std::span<const PrecomputedTri<T>> precomputed,
    QueryRay<T> ray)
{
    QueryResult<T> best;
    for (size_t i = 0; i < triangles.size(); ++i) {
        auto hit = precomputed[i].intersect(ray);
        if (!hit)
            continue;
        auto [t, u, v] = *hit;
        if (!best.hit || t < best.t || (t == best.t && i < best.prim_id)) {
            best.hit = true;
            best.prim_id = i;
            best.t = t;
            best.u = u;
            best.v = v;
            ray.tmax = t;
        }
    }
    return best;
}

template <typename T, bool IsRobust>
QueryResult<T> intersect_bvh(
    const Bvh<T>& bvh,
    std::span<const PrecomputedTri<T>> precomputed,
    QueryRay<T> ray)
{
    QueryResult<T> best;
    GrowingStack<typename Bvh<T>::Index> stack;
    bvh.template intersect<false, IsRobust>(ray, bvh.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto prim_id = bvh.prim_ids[i];
                auto hit = precomputed[prim_id].intersect(ray);
                if (!hit)
                    continue;
                auto [t, u, v] = *hit;
                if (!best.hit || t < best.t || (t == best.t && prim_id < best.prim_id)) {
                    best.hit = true;
                    best.prim_id = prim_id;
                    best.t = t;
                    best.u = u;
                    best.v = v;
                    ray.tmax = t;
                }
            }
            return best.hit;
        });
    return best;
}

template <typename T>
Bvh<T> build_bvh(
    ThreadPool& thread_pool,
    std::span<const Triangle<T>> triangles,
    typename DefaultBuilder<BvhNode<T>>::Quality quality)
{
    std::vector<TriangleBBox<T>> bboxes(triangles.size());
    std::vector<Vec3<T>> centers(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        bboxes[i] = triangles[i].get_bbox();
        centers[i] = triangles[i].get_center();
    }

    typename DefaultBuilder<BvhNode<T>>::Config config;
    config.quality = quality;
    config.parallel_threshold = 1024;
    return DefaultBuilder<BvhNode<T>>::build(thread_pool, bboxes, centers, config);
}

struct Stats {
    size_t groups = 0;
    size_t scale_groups = 0;
    std::array<size_t, 5> category_groups {};
    std::array<size_t, 5> category_primitives {};
    size_t primitives = 0;
    size_t degenerate_primitives = 0;
    size_t max_primitives = 0;
    size_t rays = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t brute_checks = 0;
    size_t brute_mismatches = 0;
    size_t build_checks = 0;
    size_t build_structure_mismatches = 0;
    size_t build_query_checks = 0;
    size_t build_query_mismatches = 0;
    size_t order_checks = 0;
    size_t order_hit_mismatches = 0;
    size_t order_t_mismatches = 0;
    size_t shuffled_hits = 0;
    size_t fast_checks = 0;
    size_t fast_mismatches = 0;
    std::vector<std::string> failures;

    void fail(std::string message) {
        if (failures.size() < 20)
            failures.push_back(std::move(message));
    }

    bool passed() const {
        return brute_mismatches == 0 &&
            build_structure_mismatches == 0 &&
            build_query_mismatches == 0 &&
            order_hit_mismatches == 0 &&
            order_t_mismatches == 0 &&
            fast_mismatches == 0;
    }

    double hit_rate() const {
        return rays == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(rays);
    }

    double shuffled_hit_rate() const {
        return rays == 0 ? 0.0 : static_cast<double>(shuffled_hits) / static_cast<double>(rays);
    }
};

template <typename T>
std::string describe_result(const QueryResult<T>& result) {
    std::ostringstream stream;
    if (!result.hit) {
        stream << "miss";
    } else {
        stream << "hit(id=" << result.prim_id
               << ", t=" << std::setprecision(std::numeric_limits<T>::max_digits10) << result.t
               << ", u=" << result.u
               << ", v=" << result.v << ')';
    }
    return stream.str();
}

template <typename T>
void run_case(Scene<T>& scene, ThreadPool& thread_pool, Stats& stats, Rng& rng) {
    const auto quality = scene.category == Category::Scale
        ? DefaultBuilder<BvhNode<T>>::Quality::Medium
        : DefaultBuilder<BvhNode<T>>::Quality::High;

    auto bvh_a = build_bvh<T>(thread_pool, scene.triangles, quality);
    auto bvh_b = build_bvh<T>(thread_pool, scene.triangles, quality);

    ++stats.build_checks;
    if (bvh_a != bvh_b) {
        ++stats.build_structure_mismatches;
        stats.fail(
            std::string(category_name(scene.category)) + " case " +
            std::to_string(scene.case_id) +
            ": two builds produced different BVHs");
    }

    std::vector<PrecomputedTri<T>> precomputed(scene.triangles.size());
    for (size_t i = 0; i < scene.triangles.size(); ++i)
        precomputed[i] = PrecomputedTri<T>(scene.triangles[i]);

    auto shuffled = scene.triangles;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    auto shuffled_bvh = build_bvh<T>(thread_pool, shuffled, quality);

    std::vector<PrecomputedTri<T>> shuffled_precomputed(shuffled.size());
    for (size_t i = 0; i < shuffled.size(); ++i)
        shuffled_precomputed[i] = PrecomputedTri<T>(shuffled[i]);

    const bool check_fast = scene.category != Category::Huge;
    for (const auto& ray : scene.rays) {
        auto brute = intersect_bruteforce<T>(scene.triangles, precomputed, ray);
        auto robust = intersect_bvh<T, true>(bvh_a, precomputed, ray);

        ++stats.rays;
        if (robust.hit)
            ++stats.hits;
        else
            ++stats.misses;

        ++stats.brute_checks;
        if (!exact_equal(brute, robust)) {
            ++stats.brute_mismatches;
            stats.fail(
                std::string(category_name(scene.category)) + " case " +
                std::to_string(scene.case_id) + " robust: brute=" +
                describe_result(brute) + ", bvh=" + describe_result(robust));
        }

        ++stats.build_query_checks;
        auto robust_b = intersect_bvh<T, true>(bvh_b, precomputed, ray);
        if (!exact_equal(robust, robust_b)) {
            ++stats.build_query_mismatches;
            stats.fail(
                std::string(category_name(scene.category)) + " case " +
                std::to_string(scene.case_id) + ": builds differ at query");
        }

        ++stats.order_checks;
        auto shuffled_result = intersect_bvh<T, true>(shuffled_bvh, shuffled_precomputed, ray);
        if (shuffled_result.hit)
            ++stats.shuffled_hits;
        if (!order_invariant_equal(robust, shuffled_result)) {
            if (robust.hit != shuffled_result.hit) {
                ++stats.order_hit_mismatches;
                stats.fail(
                    std::string(category_name(scene.category)) + " case " +
                    std::to_string(scene.case_id) + " shuffled: hit differs");
            } else {
                ++stats.order_t_mismatches;
                stats.fail(
                    std::string(category_name(scene.category)) + " case " +
                    std::to_string(scene.case_id) + " shuffled: t differs");
            }
        }

        if (check_fast) {
            ++stats.fast_checks;
            auto fast = intersect_bvh<T, false>(bvh_a, precomputed, ray);
            if (!exact_equal(brute, fast)) {
                ++stats.fast_mismatches;
                stats.fail(
                    std::string(category_name(scene.category)) + " case " +
                    std::to_string(scene.case_id) + " fast: brute=" +
                    describe_result(brute) + ", bvh=" + describe_result(fast));
            }
        }
    }

    stats.primitives += scene.triangles.size();
    stats.degenerate_primitives += scene.degenerate_count;
    stats.max_primitives = std::max(stats.max_primitives, scene.triangles.size());
    if (scene.category == Category::Scale) {
        ++stats.scale_groups;
    } else {
        ++stats.groups;
        auto category = static_cast<size_t>(scene.category);
        ++stats.category_groups[category];
        stats.category_primitives[category] += scene.triangles.size();
    }
}

struct Options {
    size_t groups = 10000;
    size_t scale_runs = 2;
    size_t scale_primitives = 100000;
    uint64_t seed = 0x3677b17a5eedULL;
    std::string report_path;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --groups <n>             Random scene groups (default: 10000)\n"
        << "  --scale-runs <n>         100k-primitive scene runs (default: 2)\n"
        << "  --scale-primitives <n>   Primitive count for scale runs (default: 100000)\n"
        << "  --seed <n>               Deterministic 64-bit seed\n"
        << "  --report <path>          Write a Markdown summary\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        auto next_value = [&] () -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << '\n';
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--groups") {
            auto value = next_value();
            if (!value)
                return false;
            options.groups = std::strtoull(value, nullptr, 10);
        } else if (arg == "--scale-runs") {
            auto value = next_value();
            if (!value)
                return false;
            options.scale_runs = std::strtoull(value, nullptr, 10);
        } else if (arg == "--scale-primitives") {
            auto value = next_value();
            if (!value)
                return false;
            options.scale_primitives = std::strtoull(value, nullptr, 10);
        } else if (arg == "--seed") {
            auto value = next_value();
            if (!value)
                return false;
            options.seed = std::strtoull(value, nullptr, 0);
        } else if (arg == "--report") {
            auto value = next_value();
            if (!value)
                return false;
            options.report_path = value;
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            return false;
        }
    }
    return true;
}

std::array<size_t, 5> group_distribution(size_t groups) {
    std::array<size_t, 5> counts {
        groups / 10,
        groups * 30 / 100,
        groups * 30 / 100,
        groups * 20 / 100,
        0
    };
    counts[4] = groups -
        counts[0] - counts[1] - counts[2] - counts[3];
    return counts;
}

template <typename T>
Stats run_suite(
    size_t groups,
    size_t scale_runs,
    size_t scale_primitives,
    uint64_t seed)
{
    Stats stats;
    Rng rng(seed);
    ThreadPool thread_pool;
    auto distribution = group_distribution(groups);

    size_t case_id = 0;
    for (size_t i = 0; i < distribution[0]; ++i) {
        auto scene = make_scene<T>(Category::Single, 1, rng, case_id++);
        scene.rays.reserve(24);
        for (size_t j = 0; j < 24; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }
    for (size_t i = 0; i < distribution[1]; ++i) {
        auto count = size_t{2} + static_cast<size_t>(rng.below(31));
        auto scene = make_scene<T>(Category::Normal, count, rng, case_id++);
        scene.rays.reserve(16);
        for (size_t j = 0; j < 16; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }
    for (size_t i = 0; i < distribution[2]; ++i) {
        auto count = size_t{1} + static_cast<size_t>(rng.below(32));
        auto scene = make_scene<T>(Category::Degenerate, count, rng, case_id++);
        scene.rays.reserve(16);
        for (size_t j = 0; j < 16; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }
    for (size_t i = 0; i < distribution[3]; ++i) {
        auto count = size_t{1} + static_cast<size_t>(rng.below(16));
        auto scene = make_scene<T>(Category::Huge, count, rng, case_id++);
        scene.rays.reserve(16);
        for (size_t j = 0; j < 16; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }
    for (size_t i = 0; i < distribution[4]; ++i) {
        auto count = size_t{2} + static_cast<size_t>(rng.below(24));
        auto scene = make_scene<T>(Category::Mixed, count, rng, case_id++);
        scene.rays.reserve(16);
        for (size_t j = 0; j < 16; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }
    for (size_t i = 0; i < scale_runs; ++i) {
        auto scene = make_scene<T>(Category::Scale, scale_primitives, rng, case_id++);
        scene.rays.reserve(12);
        for (size_t j = 0; j < 12; ++j)
            scene.rays.push_back(make_ray(scene, rng));
        run_case(scene, thread_pool, stats, rng);
    }

    return stats;
}

void print_stats(std::ostream& out, std::string_view scalar, const Stats& stats) {
    out
        << "[" << scalar << "]\n"
        << "  groups: " << stats.groups
        << ", scale groups: " << stats.scale_groups
        << ", primitives: " << stats.primitives
        << ", degenerate primitives: " << stats.degenerate_primitives
        << ", max primitives: " << stats.max_primitives << '\n'
        << "  scene coverage: single=" << stats.category_groups[0]
        << ", normal=" << stats.category_groups[1]
        << ", degenerate=" << stats.category_groups[2]
        << ", huge=" << stats.category_groups[3]
        << ", mixed=" << stats.category_groups[4] << '\n'
        << "  scene primitives: single=" << stats.category_primitives[0]
        << ", normal=" << stats.category_primitives[1]
        << ", degenerate=" << stats.category_primitives[2]
        << ", huge=" << stats.category_primitives[3]
        << ", mixed=" << stats.category_primitives[4] << '\n'
        << "  rays: " << stats.rays
        << ", hits: " << stats.hits
        << ", misses: " << stats.misses
        << ", hit rate: " << std::fixed << std::setprecision(6) << stats.hit_rate() << '\n'
        << "  brute vs robust: " << stats.brute_checks
        << " checks, " << stats.brute_mismatches << " mismatches\n"
        << "  build twice: " << stats.build_checks
        << " checks, " << stats.build_structure_mismatches << " structure mismatches, "
        << stats.build_query_checks << " query checks, "
        << stats.build_query_mismatches << " query mismatches\n"
        << "  order invariance: " << stats.order_checks
        << " checks, " << stats.order_hit_mismatches << " hit mismatches, "
        << stats.order_t_mismatches << " distance mismatches"
        << " (original hit rate " << stats.hit_rate()
        << ", shuffled hit rate " << stats.shuffled_hit_rate() << ")\n"
        << "  fast traversal: " << stats.fast_checks
        << " checks, " << stats.fast_mismatches << " mismatches\n";

    if (!stats.failures.empty()) {
        out << "  failures:\n";
        for (const auto& failure : stats.failures)
            out << "    - " << failure << '\n';
    }
}

struct TotalStats {
    size_t groups = 0;
    size_t scale_groups = 0;
    std::array<size_t, 5> category_groups {};
    std::array<size_t, 5> category_primitives {};
    size_t primitives = 0;
    size_t degenerate_primitives = 0;
    size_t max_primitives = 0;
    size_t rays = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t brute_checks = 0;
    size_t brute_mismatches = 0;
    size_t build_checks = 0;
    size_t build_structure_mismatches = 0;
    size_t build_query_checks = 0;
    size_t build_query_mismatches = 0;
    size_t order_checks = 0;
    size_t order_hit_mismatches = 0;
    size_t order_t_mismatches = 0;
    size_t shuffled_hits = 0;
    size_t fast_checks = 0;
    size_t fast_mismatches = 0;

    void add(const Stats& stats) {
        groups += stats.groups;
        scale_groups += stats.scale_groups;
        for (size_t i = 0; i < category_groups.size(); ++i) {
            category_groups[i] += stats.category_groups[i];
            category_primitives[i] += stats.category_primitives[i];
        }
        primitives += stats.primitives;
        degenerate_primitives += stats.degenerate_primitives;
        max_primitives = std::max(max_primitives, stats.max_primitives);
        rays += stats.rays;
        hits += stats.hits;
        misses += stats.misses;
        brute_checks += stats.brute_checks;
        brute_mismatches += stats.brute_mismatches;
        build_checks += stats.build_checks;
        build_structure_mismatches += stats.build_structure_mismatches;
        build_query_checks += stats.build_query_checks;
        build_query_mismatches += stats.build_query_mismatches;
        order_checks += stats.order_checks;
        order_hit_mismatches += stats.order_hit_mismatches;
        order_t_mismatches += stats.order_t_mismatches;
        shuffled_hits += stats.shuffled_hits;
        fast_checks += stats.fast_checks;
        fast_mismatches += stats.fast_mismatches;
    }

    bool passed() const {
        return brute_mismatches == 0 &&
            build_structure_mismatches == 0 &&
            build_query_mismatches == 0 &&
            order_hit_mismatches == 0 &&
            order_t_mismatches == 0 &&
            fast_mismatches == 0;
    }

    double hit_rate() const {
        return rays == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(rays);
    }

    double shuffled_hit_rate() const {
        return rays == 0 ? 0.0 : static_cast<double>(shuffled_hits) / static_cast<double>(rays);
    }
};

void write_report(
    const std::string& path,
    const Options& options,
    const Stats& float_stats,
    const Stats& double_stats,
    const TotalStats& totals,
    double seconds)
{
    std::ofstream out(path);
    out
        << "# BVH randomized property test\n\n"
        << "- Seed: `" << options.seed << "`\n"
        << "- Result: **" << (totals.passed() ? "PASS" : "FAIL") << "**\n"
        << "- Elapsed: " << std::fixed << std::setprecision(3) << seconds << " s\n"
        << "- Random groups: " << totals.groups << "\n"
        << "- Scale groups: " << totals.scale_groups
        << " at " << options.scale_primitives << " primitives\n"
        << "- Scene coverage: single=" << totals.category_groups[0]
        << ", normal=" << totals.category_groups[1]
        << ", degenerate=" << totals.category_groups[2]
        << ", huge=" << totals.category_groups[3]
        << ", mixed=" << totals.category_groups[4] << "\n"
        << "- Scene primitives: single=" << totals.category_primitives[0]
        << ", normal=" << totals.category_primitives[1]
        << ", degenerate=" << totals.category_primitives[2]
        << ", huge=" << totals.category_primitives[3]
        << ", mixed=" << totals.category_primitives[4] << "\n"
        << "- Total primitives: " << totals.primitives
        << " (degenerate: " << totals.degenerate_primitives << ")\n"
        << "- Rays: " << totals.rays
        << " (hits: " << totals.hits << ", misses: " << totals.misses
        << ", hit rate: " << std::setprecision(6) << totals.hit_rate() << ")\n\n"
        << "| Property | Checks | Mismatches |\n"
        << "|---|---:|---:|\n"
        << "| brute force vs robust BVH | " << totals.brute_checks << " | "
        << totals.brute_mismatches << " |\n"
        << "| two builds, exact BVH equality | " << totals.build_checks << " | "
        << totals.build_structure_mismatches << " |\n"
        << "| two builds, query equality | " << totals.build_query_checks << " | "
        << totals.build_query_mismatches << " |\n"
        << "| primitive-order hit invariance | " << totals.order_checks << " | "
        << totals.order_hit_mismatches << " |\n"
        << "| primitive-order distance invariance | " << totals.order_checks << " | "
        << totals.order_t_mismatches << " |\n"
        << "| brute force vs fast BVH | " << totals.fast_checks << " | "
        << totals.fast_mismatches << " |\n\n"
        << "Original hit rate: " << std::setprecision(6) << totals.hit_rate() << "\n\n"
        << "Shuffled hit rate: " << std::setprecision(6) << totals.shuffled_hit_rate() << "\n\n"
        << "## By scalar\n\n";
    print_stats(out, "float", float_stats);
    out << '\n';
    print_stats(out, "double", double_stats);

    if (!float_stats.failures.empty() || !double_stats.failures.empty()) {
        out << "\n## Failures\n\n";
        for (const auto& failure : float_stats.failures)
            out << "- float: " << failure << '\n';
        for (const auto& failure : double_stats.failures)
            out << "- double: " << failure << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options))
        return 0;

    auto start = std::chrono::steady_clock::now();
    auto float_groups = options.groups / 2;
    auto double_groups = options.groups - float_groups;
    auto float_scale_runs = options.scale_runs / 2;
    auto double_scale_runs = options.scale_runs - float_scale_runs;

    auto float_stats = run_suite<float>(
        float_groups, float_scale_runs, options.scale_primitives,
        options.seed ^ 0xf10a7ULL);
    auto double_stats = run_suite<double>(
        double_groups, double_scale_runs, options.scale_primitives,
        options.seed ^ 0xd0b1eULL);

    TotalStats totals;
    totals.add(float_stats);
    totals.add(double_stats);

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout
        << "BVH randomized property test\n"
        << "seed=" << options.seed
        << ", random groups=" << options.groups
        << ", scale groups=" << options.scale_runs
        << " x " << options.scale_primitives << " primitives"
        << ", elapsed=" << std::fixed << std::setprecision(3) << seconds << " s\n\n";
    print_stats(std::cout, "float", float_stats);
    std::cout << '\n';
    print_stats(std::cout, "double", double_stats);
    std::cout
        << "\n[total]\n"
        << "  groups: " << totals.groups
        << ", scale groups: " << totals.scale_groups
        << ", primitives: " << totals.primitives
        << ", degenerate primitives: " << totals.degenerate_primitives
        << ", max primitives: " << totals.max_primitives << '\n'
        << "  scene coverage: single=" << totals.category_groups[0]
        << ", normal=" << totals.category_groups[1]
        << ", degenerate=" << totals.category_groups[2]
        << ", huge=" << totals.category_groups[3]
        << ", mixed=" << totals.category_groups[4] << '\n'
        << "  scene primitives: single=" << totals.category_primitives[0]
        << ", normal=" << totals.category_primitives[1]
        << ", degenerate=" << totals.category_primitives[2]
        << ", huge=" << totals.category_primitives[3]
        << ", mixed=" << totals.category_primitives[4] << '\n'
        << "  rays: " << totals.rays
        << ", hits: " << totals.hits
        << ", misses: " << totals.misses
        << ", hit rate: " << std::setprecision(6) << totals.hit_rate() << '\n'
        << "  brute vs robust: " << totals.brute_checks
        << " checks, " << totals.brute_mismatches << " mismatches\n"
        << "  build twice: " << totals.build_checks
        << " checks, " << totals.build_structure_mismatches << " structure mismatches, "
        << totals.build_query_checks << " query checks, "
        << totals.build_query_mismatches << " query mismatches\n"
        << "  order invariance: " << totals.order_checks
        << " checks, " << totals.order_hit_mismatches << " hit mismatches, "
        << totals.order_t_mismatches << " distance mismatches"
        << " (original hit rate " << totals.hit_rate()
        << ", shuffled hit rate " << totals.shuffled_hit_rate() << ")\n"
        << "  fast traversal: " << totals.fast_checks
        << " checks, " << totals.fast_mismatches << " mismatches\n"
        << "  result: " << (totals.passed() ? "PASS" : "FAIL") << '\n';

    if (!options.report_path.empty())
        write_report(options.report_path, options, float_stats, double_stats, totals, seconds);

    return totals.passed() ? 0 : 1;
}
