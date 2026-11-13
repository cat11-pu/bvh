#ifndef BVH_V2_BBOX_H
#define BVH_V2_BBOX_H

#include "bvh/v2/vec.h"
#include "bvh/v2/utils.h"
#include "bvh/v2/ray.h"

#include <cmath>
#include <limits>
#include <utility>

namespace bvh::v2 {

template <typename T, size_t N>
struct BBox {
    Vec<T, N> min, max;

    BBox() = default;
    BVH_ALWAYS_INLINE BBox(const Vec<T, N>& min, const Vec<T, N>& max) : min(min), max(max) {}
    BVH_ALWAYS_INLINE explicit BBox(const Vec<T, N>& point) : BBox(point, point) {}

    BVH_ALWAYS_INLINE BBox& extend(const Vec<T, N>& point) {
        return extend(BBox(point));
    }

    BVH_ALWAYS_INLINE BBox& extend(const BBox& other) {
        min = robust_min(min, other.min);
        max = robust_max(max, other.max);
        return *this;
    }

    BVH_ALWAYS_INLINE Vec<T, N> get_diagonal() const { return max - min; }
    BVH_ALWAYS_INLINE Vec<T, N> get_center() const { return (max + min) * static_cast<T>(0.5); }

    /// Slab-based ray/box intersection test. Returns the entry and exit distances along the ray;
    /// when the ray misses the box, the entry distance is strictly greater than the exit distance.
    /// This is the same test used for the leaf-level expanded boxes so that a brute-force loop
    /// over the expanded boxes matches BVH traversal semantics exactly.
    [[nodiscard]] BVH_ALWAYS_INLINE std::pair<T, T> intersect_ray(const Ray<T, N>& ray) const {
        T t0 = ray.tmin;
        T t1 = ray.tmax;
        static_for<0, N>([&] (size_t i) {
            T inv_dir = safe_inverse(ray.dir[i]);
            T near = (min[i] - ray.org[i]) * inv_dir;
            T far  = (max[i] - ray.org[i]) * inv_dir;
            if (far < near)
                std::swap(near, far);
            t0 = robust_max(t0, near);
            t1 = robust_min(t1, far);
        });
        return std::pair<T, T> { t0, t1 };
    }

    /// Numerically robust ray/box test matching `Node::intersect_robust()` (T. Ize's
    /// "Robust BVH Ray Traversal"): the exit slabs use an inverse direction padded by 2 ULPs so
    /// that a ray touching a box boundary is never wrongly culled. Use this on the expanded
    /// primitive boxes in a brute-force reference loop to stay bit-exact with robust BVH
    /// traversal.
    [[nodiscard]] BVH_ALWAYS_INLINE std::pair<T, T> intersect_ray_robust(const Ray<T, N>& ray) const {
        auto inv_dir = ray.template get_inv_dir<false>();
        auto inv_dir_pad = Ray<T, N>::pad_inv_dir(inv_dir);
        auto octant = ray.get_octant();
        T t0 = ray.tmin;
        T t1 = ray.tmax;
        static_for<0, N>([&] (size_t i) {
            T t_near = (octant[i] ? max[i] : min[i]) - ray.org[i];
            T t_far  = (octant[i] ? min[i] : max[i]) - ray.org[i];
            t0 = robust_max(t0, t_near * inv_dir[i]);
            t1 = robust_min(t1, t_far * inv_dir_pad[i]);
        });
        return std::pair<T, T> { t0, t1 };
    }

    /// Grows this bounding box by the given per-axis margin on every side. This is the building
    /// block for custom primitive bounds such as the swept boxes used in motion blur rendering:
    /// a BVH built with grown bounds is only correct if the traversal uses the exact same grown
    /// bounds. When `robust` is true, the bounds are additionally pushed outward by one ULP, which
    /// guarantees that the box stays a conservative superset even after floating-point rounding.
    BVH_ALWAYS_INLINE BBox& grow(const Vec<T, N>& margin, bool robust = false) {
        static_for<0, N>([&] (size_t i) {
            min[i] -= margin[i];
            max[i] += margin[i];
            if (robust && std::isfinite(min[i]))
                min[i] = std::nextafter(min[i], -std::numeric_limits<T>::infinity());
            if (robust && std::isfinite(max[i]))
                max[i] = std::nextafter(max[i], +std::numeric_limits<T>::infinity());
        });
        return *this;
    }

    /// Grows this bounding box by the given (uniform) margin on every side and axis.
    BVH_ALWAYS_INLINE BBox& grow(T margin, bool robust = false) {
        return grow(Vec<T, N>(margin), robust);
    }

    BVH_ALWAYS_INLINE T get_half_area() const {
        auto d = get_diagonal();
        static_assert(N == 2 || N == 3);
        if constexpr (N == 3) return (d[0] + d[1]) * d[2] + d[0] * d[1];
        if constexpr (N == 2) return d[0] + d[1];
        return static_cast<T>(0.);
    }

    BVH_ALWAYS_INLINE static constexpr BBox make_empty() {
        return BBox(
            Vec<T, N>(+std::numeric_limits<T>::max()),
            Vec<T, N>(-std::numeric_limits<T>::max()));
    }
};

} // namespace bvh::v2

#endif
