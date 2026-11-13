#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/bbox.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/motion_blur.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
static constexpr size_t stack_size = 64;

struct HitResult {
    size_t prim_id = invalid_id;
    Scalar t = std::numeric_limits<Scalar>::max();
    Scalar u = 0, v = 0;

    bool operator == (const HitResult& other) const {
        if (prim_id != other.prim_id)
            return false;
        if (prim_id == invalid_id)
            return true;
        return t == other.t && u == other.u && v == other.v;
    }
};

struct TraversalStats {
    uint64_t box_tests = 0;
    uint64_t leaf_visits = 0;
    uint64_t prim_tests = 0;
};

struct Rng {
    std::mt19937_64 engine;
    std::uniform_real_distribution<Scalar> dist;

    Rng(uint64_t seed, Scalar lo, Scalar hi)
        : engine(seed), dist(lo, hi)
    {}

    Scalar next() { return dist(engine); }
};

static Vec3 random_point(Rng& rng) {
    return Vec3(rng.next(), rng.next(), rng.next());
}

static Vec3 random_unit(Rng& rng) {
    Vec3 d(rng.next() * 2 - 1, rng.next() * 2 - 1, rng.next() * 2 - 1);
    Scalar length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    return d * (Scalar(1) / (length + std::numeric_limits<Scalar>::min()));
}

/// A scene of random moving triangles. Each primitive has a position at shutter open (`tri_t0`)
/// and a motion vector giving its displacement at shutter close, plus a non-negative custom
/// expansion margin modeling application-level bounds (primitive thickness, blur padding, ...).
struct MotionScene {
    struct Primitive {
        Tri tri_t0;
        Vec3 motion;
        Vec3 margin;
    };

    std::vector<Primitive> primitives;

    explicit MotionScene(size_t prim_count, uint64_t seed) {
        Rng rng(seed, Scalar(-10), Scalar(10));
        std::mt19937_64 margin_engine(seed * 1000003 + 7);
        std::uniform_real_distribution<Scalar> margin_dist(Scalar(0), Scalar(0.05));
        primitives.reserve(prim_count);
        for (size_t i = 0; i < prim_count; ++i) {
            Vec3 center = random_point(rng) * Scalar(2);
            Scalar size = Scalar(0.1) + rng.next() * Scalar(0.4);
            Tri tri(
                center + random_unit(rng) * size,
                center + random_unit(rng) * size,
                center + random_unit(rng) * size);

            // Motion amplitude is comparable to the primitive size so swept boxes overlap their
            // static boxes substantially, exercising both pruning and false positives.
            Scalar motion_scale = rng.next() * Scalar(1.5);
            Vec3 motion = random_unit(rng) * motion_scale;

            // Margins must stay non-negative: a negative margin would shrink the swept box below
            // the primitive bounds and let a node test cull a real hit.
            Vec3 margin(margin_dist(margin_engine) + Scalar(1e-4));
            primitives.push_back({ tri, motion, margin });
        }
    }

    BBox bbox_t0(size_t i) const { return primitives[i].tri_t0.get_bbox(); }

    BBox bbox_t1(size_t i) const {
        const auto& p = primitives[i];
        return Tri(
            p.tri_t0.p0 + p.motion,
            p.tri_t0.p1 + p.motion,
            p.tri_t0.p2 + p.motion).get_bbox();
    }
};

/// Brute-force reference traversal. It runs the exact same ray/triangle test a BVH leaf callback
/// runs, but over every primitive and with no bounding-volume culling. Expanded boxes are pure
/// acceleration data: correctness of the BVH only requires that its node boxes be a conservative
/// superset of the primitives, so an un-culled triangle loop is the natural oracle.
static HitResult intersect_brute_force(
    const MotionScene& scene,
    const std::vector<PrecomputedTri>& tris_t0,
    Ray ray,
    TraversalStats& stats)
{
    HitResult best;
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        ++stats.prim_tests;
        if (auto hit = tris_t0[i].intersect(ray)) {
            best.prim_id = i;
            std::tie(best.t, best.u, best.v) = *hit;
            ray.tmax = best.t;
        }
    }
    return best;
}

int main() {
    static constexpr size_t prim_count = 4096;
    static constexpr size_t ray_count  = 20000;
    static constexpr size_t seed_count = 4;

    using Quality = bvh::v2::DefaultBuilder<Node>::Quality;
    const char* quality_env = std::getenv("BVH_QUALITY");
    const bool use_robust = std::getenv("BVH_FAST_TRAVERSAL") == nullptr;
    Quality quality = Quality::High;
    if (quality_env) {
        std::string q = quality_env;
        if (q == "low") quality = Quality::Low;
        else if (q == "med") quality = Quality::Medium;
    }

    std::cout
        << "builder quality:  " << (quality == Quality::Low ? "low"
            : quality == Quality::Medium ? "medium" : "high") << "\n"
        << "robust traversal: " << (use_robust ? "yes" : "no") << "\n";

    uint64_t total_mismatches = 0;
    uint64_t total_rays = 0;
    uint64_t total_hits = 0;
    TraversalStats bvh_stats, brute_stats;

    std::cout
        << "scene primitives: " << prim_count << "\n"
        << "rays per scene:   " << ray_count << "\n"
        << "scenes:           " << seed_count << "\n" << std::endl;

    for (uint64_t seed = 1; seed <= seed_count; ++seed) {
        MotionScene scene(prim_count, seed);

        std::vector<BBox> bboxes_t0(prim_count), bboxes_t1(prim_count);
        std::vector<Vec3> margins(prim_count);
        for (size_t i = 0; i < prim_count; ++i) {
            bboxes_t0[i] = scene.bbox_t0(i);
            bboxes_t1[i] = scene.bbox_t1(i);
            margins[i]   = scene.primitives[i].margin;
        }

        // Expanded boxes: union(t0, t1) + custom margin + 1-ULP outward padding. The same boxes
        // drive both the build and the traversal (node bounds are unions of them).
        std::vector<BBox> expanded_bboxes(prim_count);
        std::vector<Vec3> expanded_centers(prim_count);
        bvh::v2::MotionBlurBoxes<Scalar, 3>::build(
            bboxes_t0, bboxes_t1, expanded_bboxes, expanded_centers, margins, true);

        // Build-time invariant: every expanded box must contain both endpoint boxes.
        for (size_t i = 0; i < prim_count; ++i) {
            for (size_t axis = 0; axis < 3; ++axis) {
                assert(expanded_bboxes[i].min[axis] <= bboxes_t0[i].min[axis]);
                assert(expanded_bboxes[i].max[axis] >= bboxes_t0[i].max[axis]);
                assert(expanded_bboxes[i].min[axis] <= bboxes_t1[i].min[axis]);
                assert(expanded_bboxes[i].max[axis] >= bboxes_t1[i].max[axis]);
            }
        }

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = quality;
        Bvh bvh = bvh::v2::DefaultBuilder<Node>::build(expanded_bboxes, expanded_centers, config);

        std::vector<PrecomputedTri> tris_t0(prim_count);
        for (size_t i = 0; i < prim_count; ++i)
            tris_t0[i] = PrecomputedTri(scene.primitives[i].tri_t0);

        // Permute reference data in BVH leaf order for direct indexing.
        std::vector<PrecomputedTri> permuted_tris(prim_count);
        for (size_t i = 0; i < prim_count; ++i)
            permuted_tris[i] = tris_t0[bvh.prim_ids[i]];

        Rng ray_rng(seed * 7919 + 13, Scalar(-25), Scalar(25));
        uint64_t mismatches = 0;
        uint64_t hits = 0;

        for (size_t r = 0; r < ray_count; ++r) {
            Ray ray(
                random_point(ray_rng) * Scalar(2),
                random_unit(ray_rng),
                Scalar(0),
                Scalar(1e3));

            // Reference brute-force result (no box culling).
            Ray brute_ray = ray;
            TraversalStats brute_local;
            HitResult brute_hit = intersect_brute_force(scene, tris_t0, brute_ray, brute_local);

            // BVH traversal over the expanded boxes. Robust node tests avoid grazing-ray culls.
            Ray bvh_ray = ray;
            HitResult bvh_hit;
            TraversalStats bvh_local;
            bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
            auto leaf_fn = [&] (size_t begin, size_t end) {
                ++bvh_local.leaf_visits;
                bool found = false;
                for (size_t i = begin; i < end; ++i) {
                    ++bvh_local.prim_tests;
                    if (auto hit = permuted_tris[i].intersect(bvh_ray)) {
                        bvh_hit.prim_id = bvh.prim_ids[i];
                        std::tie(bvh_ray.tmax, bvh_hit.u, bvh_hit.v) = *hit;
                        bvh_hit.t = bvh_ray.tmax;
                        found = true;
                    }
                }
                return found;
            };
            auto inner_fn = [&] (const Node&, const Node&) {
                bvh_local.box_tests += 2;
            };
            if (use_robust)
                bvh.intersect<false, true>(bvh_ray, bvh.get_root().index, stack, leaf_fn, inner_fn);
            else
                bvh.intersect<false, false>(bvh_ray, bvh.get_root().index, stack, leaf_fn, inner_fn);

            if (!(bvh_hit == brute_hit))
                ++mismatches;
            if (brute_hit.prim_id != invalid_id)
                ++hits;

            bvh_stats.box_tests    += bvh_local.box_tests;
            bvh_stats.leaf_visits  += bvh_local.leaf_visits;
            bvh_stats.prim_tests   += bvh_local.prim_tests;
            brute_stats.prim_tests += brute_local.prim_tests;
        }

        total_mismatches += mismatches;
        total_rays += ray_count;
        total_hits += hits;

        std::cout
            << "scene " << seed << ": " << mismatches << " mismatches / " << ray_count
            << " rays (" << hits << " hits)" << std::endl;
    }

    auto avg = [] (uint64_t total, uint64_t count) {
        return static_cast<double>(total) / static_cast<double>(count);
    };

    std::cout << "\n=== consistency ===" << std::endl;
    std::cout << "total rays:        " << total_rays << std::endl;
    std::cout << "total hits:        " << total_hits << std::endl;
    std::cout << "mismatches:        " << total_mismatches << std::endl;

    std::cout << "\n=== average work per ray ===" << std::endl;
    std::cout << "brute-force triangle tests: " << avg(brute_stats.prim_tests, total_rays)
              << " (= " << prim_count << " per ray)" << std::endl;
    std::cout << "bvh node box tests:         " << avg(bvh_stats.box_tests, total_rays) << std::endl;
    std::cout << "bvh leaf visits:            " << avg(bvh_stats.leaf_visits, total_rays) << std::endl;
    std::cout << "bvh triangle tests:         " << avg(bvh_stats.prim_tests, total_rays) << std::endl;
    std::cout << "triangle-test speedup:      "
              << avg(brute_stats.prim_tests, total_rays) /
                     (avg(bvh_stats.prim_tests, total_rays) + 1e-9)
              << "x" << std::endl;

    return total_mismatches == 0 ? 0 : 1;
}
