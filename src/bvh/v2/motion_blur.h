#ifndef BVH_V2_MOTION_BLUR_H
#define BVH_V2_MOTION_BLUR_H

#include "bvh/v2/bbox.h"
#include "bvh/v2/vec.h"

#include <span>
#include <cassert>
#include <cstddef>

namespace bvh::v2 {

/// Helpers to build conservative, per-primitive bounding boxes for motion blur. The idea is to
/// replace each primitive bounds with the union of the bounds at shutter open and shutter close
/// (the swept volume), optionally padded by a custom margin and grown outward by one ULP so that
/// the box remains a strict superset regardless of floating-point rounding.
///
/// Correctness requirement: the very same expanded boxes must be fed both to the BVH builder and
/// to the traversal code (node bounds are derived from them), otherwise ray-primitive hits can be
/// culled by a node test and the result will differ from a brute-force traversal.
template <typename T, size_t N>
struct MotionBlurBoxes {
    using Vec  = bvh::v2::Vec<T, N>;
    using BBox = bvh::v2::BBox<T, N>;

    /// Returns the bounds of a primitive moving from `bbox_t0` to `bbox_t1` over the shutter
    /// interval. The two boxes are joined, then grown by `margin` on every side.
    [[nodiscard]] BVH_ALWAYS_INLINE static BBox make_swept_bbox(
        const BBox& bbox_t0,
        const BBox& bbox_t1,
        const Vec& margin = Vec(T(0)),
        bool robust = true)
    {
        BBox swept(bbox_t0);
        swept.extend(bbox_t1);
        swept.grow(margin, robust);
        return swept;
    }

    /// Builds expanded bounding boxes (and matching centers) for a whole batch of moving
    /// primitives. `bboxes_t0` and `bboxes_t1` contain the primitive bounds at shutter open and
    /// close respectively; `margins` optionally holds an extra per-primitive, per-axis padding.
    /// The output spans must be at least as large as the input spans.
    static void build(
        std::span<const BBox> bboxes_t0,
        std::span<const BBox> bboxes_t1,
        std::span<BBox> expanded_bboxes,
        std::span<Vec> expanded_centers,
        std::span<const Vec> margins = {},
        bool robust = true)
    {
        const size_t prim_count = bboxes_t0.size();
        assert(bboxes_t1.size() == prim_count);
        assert(expanded_bboxes.size() >= prim_count);
        assert(expanded_centers.size() >= prim_count);
        assert(margins.empty() || margins.size() >= prim_count);
        for (size_t i = 0; i < prim_count; ++i) {
            const auto margin = margins.empty() ? Vec(T(0)) : margins[i];
            expanded_bboxes[i] = make_swept_bbox(bboxes_t0[i], bboxes_t1[i], margin, robust);
            expanded_centers[i] = expanded_bboxes[i].get_center();
        }
    }
};

} // namespace bvh::v2

#endif
