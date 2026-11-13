#ifndef BVH_V2_VALIDATE_H
#define BVH_V2_VALIDATE_H

#include "bvh/v2/vec.h"
#include "bvh/v2/bbox.h"
#include "bvh/v2/tri.h"
#include "bvh/v2/bvh.h"
#include "bvh/v2/node.h"
#include "bvh/v2/default_builder.h"
#include "bvh/v2/thread_pool.h"

#include <cstddef>
#include <span>
#include <vector>

namespace bvh::v2 {

/// Returns true when the given triangle is degenerate, i.e. when it has no (positive) area.
/// This covers repeated vertices (point triangles) and collinear vertices (line segments).
template <typename T, size_t N>
BVH_ALWAYS_INLINE bool is_degenerate(const Tri<T, N>& tri) {
    auto e0 = tri.p1 - tri.p0;
    auto e1 = tri.p2 - tri.p0;
    if constexpr (N == 2) {
        // In two dimensions the (signed) doubled area is the cross product of the edges.
        return e0[0] * e1[1] - e0[1] * e1[0] == static_cast<T>(0.);
    } else {
        return length(cross(e0, e1)) == static_cast<T>(0.);
    }
}

/// Result of filtering out the degenerate primitives of a scene.
template <typename Node>
struct ValidatedScene {
    using Scalar = typename Node::Scalar;
    static constexpr size_t dimension = Node::dimension;
    using VecT  = bvh::v2::Vec<Scalar, dimension>;
    using BBoxT = bvh::v2::BBox<Scalar, dimension>;

    /// Bounding boxes of the non-degenerate primitives, in their original relative order.
    std::vector<BBoxT> bboxes;

    /// Centers of the non-degenerate primitives, in their original relative order.
    std::vector<VecT> centers;

    /// Maps an index in `bboxes`/`centers` back to the index of the primitive in the input.
    std::vector<size_t> valid_ids;

    /// Number of degenerate primitives that were skipped.
    size_t degenerate_count = 0;
};

/// Removes the degenerate (zero-area) triangles from the given scene and computes the bounding
/// boxes and centers required to build a BVH for the remaining primitives. Degenerate primitives
/// are skipped and counted rather than fed to the builders, whose grid/bin partitioning relies
/// on bounding boxes with a non-zero diagonal.
template <typename Node, typename TriScalar = typename Node::Scalar>
[[nodiscard]] ValidatedScene<Node> validate_triangles(std::span<const Tri<TriScalar, Node::dimension>> tris) {
    using Scalar = typename Node::Scalar;
    constexpr size_t dimension = Node::dimension;
    using VecT  = bvh::v2::Vec<Scalar, dimension>;

    ValidatedScene<Node> scene;
    scene.bboxes.reserve(tris.size());
    scene.centers.reserve(tris.size());
    scene.valid_ids.reserve(tris.size());

    for (size_t i = 0; i < tris.size(); ++i) {
        if (is_degenerate(tris[i])) {
            scene.degenerate_count++;
            continue;
        }
        auto bbox = tris[i].get_bbox();
        auto center = tris[i].get_center();
        scene.bboxes.emplace_back(
            VecT::generate([&] (size_t axis) { return static_cast<Scalar>(bbox.min[axis]); }),
            VecT::generate([&] (size_t axis) { return static_cast<Scalar>(bbox.max[axis]); }));
        scene.centers.emplace_back(
            VecT::generate([&] (size_t axis) { return static_cast<Scalar>(center[axis]); }));
        scene.valid_ids.push_back(i);
    }
    return scene;
}

/// Builds a BVH for the given triangles, skipping (and counting) the degenerate ones. The number
/// of skipped primitives is written to `degenerate_count` when the pointer is not null.
template <typename Node, typename TriScalar = typename Node::Scalar>
[[nodiscard]] Bvh<Node> build_validated(
    std::span<const Tri<TriScalar, Node::dimension>> tris,
    const typename DefaultBuilder<Node>::Config& config = {},
    size_t* degenerate_count = nullptr)
{
    auto scene = validate_triangles<Node>(tris);
    if (degenerate_count)
        *degenerate_count = scene.degenerate_count;
    return DefaultBuilder<Node>::build(scene.bboxes, scene.centers, config);
}

/// Same as above, but uses the given thread pool to build the BVH in parallel.
template <typename Node, typename TriScalar = typename Node::Scalar>
[[nodiscard]] Bvh<Node> build_validated(
    ThreadPool& thread_pool,
    std::span<const Tri<TriScalar, Node::dimension>> tris,
    const typename DefaultBuilder<Node>::Config& config = {},
    size_t* degenerate_count = nullptr)
{
    auto scene = validate_triangles<Node>(tris);
    if (degenerate_count)
        *degenerate_count = scene.degenerate_count;
    return DefaultBuilder<Node>::build(thread_pool, scene.bboxes, scene.centers, config);
}

} // namespace bvh::v2

#endif
