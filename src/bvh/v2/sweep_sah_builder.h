#ifndef BVH_V2_SWEEP_SAH_BUILDER_H
#define BVH_V2_SWEEP_SAH_BUILDER_H

#include "bvh/v2/top_down_sah_builder.h"
#include "bvh/v2/executor.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Single-threaded top-down builder that partitions primitives based on the Surface
/// Area Heuristic (SAH). Primitives are only sorted once along each axis.
template <typename Node>
class SweepSahBuilder : public TopDownSahBuilder<Node> {
    using typename TopDownSahBuilder<Node>::Scalar;
    using typename TopDownSahBuilder<Node>::Vec;
    using typename TopDownSahBuilder<Node>::BBox;

    using TopDownSahBuilder<Node>::build;
    using TopDownSahBuilder<Node>::config_;
    using TopDownSahBuilder<Node>::bboxes_;
    using TopDownSahBuilder<Node>::centers_;
    using TopDownSahBuilder<Node>::compute_bbox;
    using TopDownSahBuilder<Node>::try_split;
    using typename TopDownSahBuilder<Node>::WorkItem;

public:
    using typename TopDownSahBuilder<Node>::Config;

    [[nodiscard]] BVH_ALWAYS_INLINE static Bvh<Node> build(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        return SweepSahBuilder(bboxes, centers, config).build();
    }

    /// Builds a BVH in parallel using the given executor. Performs exact top-down splits
    /// until `max_subtree_count` independent subtrees are available, then builds each
    /// subtree serially in parallel tasks. This produces a BVH of the same quality as
    /// the serial builder (up to tie-breaking differences).
    template <typename Derived>
    [[nodiscard]] BVH_ALWAYS_INLINE static Bvh<Node> build_parallel(
        Executor<Derived>& executor,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config,
        size_t max_subtree_count)
    {
        return SweepSahBuilder(bboxes, centers, config, executor).build_parallel(executor, max_subtree_count);
    }

protected:
    struct Split {
        size_t pos;
        Scalar cost;
        size_t axis;
    };

    std::vector<bool> marks_;
    std::vector<Scalar> accum_;
    std::vector<size_t> prim_ids_[Node::dimension];

    BVH_ALWAYS_INLINE SweepSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : TopDownSahBuilder<Node>(bboxes, centers, config)
    {
        marks_.resize(bboxes.size());
        accum_.resize(bboxes.size());
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            prim_ids_[axis].resize(bboxes.size());
            std::iota(prim_ids_[axis].begin(), prim_ids_[axis].end(), 0);
            std::sort(prim_ids_[axis].begin(), prim_ids_[axis].end(), [&] (size_t i, size_t j) {
                return centers[i][axis] < centers[j][axis];
            });
        }
    }

    template <typename Derived>
    BVH_ALWAYS_INLINE SweepSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config,
        Executor<Derived>& executor)
        : TopDownSahBuilder<Node>(bboxes, centers, config)
    {
        marks_.resize(bboxes.size());
        accum_.resize(bboxes.size());
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            prim_ids_[axis].resize(bboxes.size());
            std::iota(prim_ids_[axis].begin(), prim_ids_[axis].end(), 0);
        }
        // Sort each axis in parallel
        executor.for_each(size_t{0}, size_t{Node::dimension}, [&] (size_t begin, size_t end) {
            for (size_t axis = begin; axis < end; ++axis) {
                std::sort(prim_ids_[axis].begin(), prim_ids_[axis].end(), [&] (size_t i, size_t j) {
                    return centers[i][axis] < centers[j][axis];
                });
            }
        });
    }

    BVH_ALWAYS_INLINE SweepSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        std::vector<size_t>&& prim_ids,
        const Config& config)
        : TopDownSahBuilder<Node>(bboxes, centers, config)
    {
        marks_.resize(bboxes.size());
        accum_.resize(prim_ids.size());
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            prim_ids_[axis] = prim_ids;
            std::sort(prim_ids_[axis].begin(), prim_ids_[axis].end(), [&] (size_t i, size_t j) {
                return centers[i][axis] < centers[j][axis];
            });
        }
    }

    template <typename Derived>
    Bvh<Node> build_parallel(Executor<Derived>& executor, size_t max_subtree_count) {
        const auto prim_count = get_prim_ids().size();
        const auto min_subtree_size = std::max(
            config_.min_leaf_size,
            (prim_count + max_subtree_count - 1) / max_subtree_count);

        // Perform exact top-down splits until enough independent subtrees are available
        Bvh<Node> top_bvh;
        top_bvh.nodes.emplace_back();
        top_bvh.nodes.back().set_bbox(compute_bbox(0, prim_count));

        std::vector<std::pair<size_t, size_t>> subtree_ranges;
        std::stack<WorkItem> stack;
        stack.push(WorkItem { 0, 0, prim_count });
        while (!stack.empty()) {
            auto item = stack.top();
            stack.pop();

            auto& node = top_bvh.nodes[item.node_id];
            if (item.size() > min_subtree_size) {
                if (auto split_pos = try_split(node.get_bbox(), item.begin, item.end)) {
                    auto first_child = top_bvh.nodes.size();
                    node.index = Node::Index::make_inner(first_child);

                    top_bvh.nodes.resize(first_child + 2);

                    auto first_bbox   = compute_bbox(item.begin, *split_pos);
                    auto second_bbox  = compute_bbox(*split_pos, item.end);
                    auto first_range  = std::make_pair(item.begin, *split_pos);
                    auto second_range = std::make_pair(*split_pos, item.end);

                    if (first_bbox.get_half_area() < second_bbox.get_half_area()) {
                        std::swap(first_bbox, second_bbox);
                        std::swap(first_range, second_range);
                    }

                    auto first_item  = WorkItem { first_child + 0, first_range.first, first_range.second };
                    auto second_item = WorkItem { first_child + 1, second_range.first, second_range.second };
                    top_bvh.nodes[first_child + 0].set_bbox(first_bbox);
                    top_bvh.nodes[first_child + 1].set_bbox(second_bbox);

                    if (first_item.size() < second_item.size())
                        std::swap(first_item, second_item);

                    stack.push(first_item);
                    stack.push(second_item);
                    continue;
                }
            }

            auto subtree_id = subtree_ranges.size();
            subtree_ranges.emplace_back(item.begin, item.end);
            node.index = Node::Index::make_leaf(subtree_id, 1);
        }

        // Build every subtree serially, in parallel tasks
        std::vector<Bvh<Node>> sub_bvhs(subtree_ranges.size());
        executor.for_each(0, subtree_ranges.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto [first, last] = subtree_ranges[i];
                std::vector<size_t> prim_ids(prim_ids_[0].begin() + first, prim_ids_[0].begin() + last);
                sub_bvhs[i] = SweepSahBuilder(bboxes_, centers_, std::move(prim_ids), config_).build();
            }
        });

        // Assemble the top-level BVH and the subtrees into the final BVH
        std::vector<size_t> node_offsets(sub_bvhs.size());
        std::vector<size_t> prim_offsets(sub_bvhs.size());
        size_t node_count = top_bvh.nodes.size();
        size_t prim_count_total = 0;
        for (size_t i = 0; i < sub_bvhs.size(); ++i) {
            node_offsets[i] = node_count - 1; // Skip root node
            prim_offsets[i] = prim_count_total;
            node_count += sub_bvhs[i].nodes.size() - 1; // idem
            prim_count_total += sub_bvhs[i].prim_ids.size();
        }

        auto copy_node = [&] (size_t i, Node& dst_node, const Node& src_node) {
            dst_node = src_node;
            dst_node.index.set_first_id(dst_node.index.first_id() +
                (src_node.is_leaf() ? prim_offsets[i] : node_offsets[i]));
        };

        top_bvh.prim_ids.resize(sub_bvhs.size());
        std::iota(top_bvh.prim_ids.begin(), top_bvh.prim_ids.end(), 0);
        for (auto& node : top_bvh.nodes) {
            if (!node.is_leaf())
                continue;
            size_t tree_id = top_bvh.prim_ids[node.index.first_id()];
            copy_node(tree_id, node, sub_bvhs[tree_id].get_root());
        }

        top_bvh.nodes.resize(node_count);
        top_bvh.prim_ids.resize(prim_count_total);
        executor.for_each(0, sub_bvhs.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto& sub_bvh = sub_bvhs[i];
                for (size_t j = 1; j < sub_bvh.nodes.size(); ++j)
                    copy_node(i, top_bvh.nodes[node_offsets[i] + j], sub_bvh.nodes[j]);
                std::copy(
                    sub_bvh.prim_ids.begin(),
                    sub_bvh.prim_ids.end(),
                    top_bvh.prim_ids.begin() + prim_offsets[i]);
            }
        });

        return top_bvh;
    }

    std::vector<size_t>& get_prim_ids() override { return prim_ids_[0]; }

    void find_best_split(size_t axis, size_t begin, size_t end, Split& best_split) {
        size_t first_right = begin;

        // Sweep from the right to the left, computing the partial SAH cost
        auto right_bbox = BBox::make_empty();
        for (size_t i = end - 1; i > begin;) {
            static constexpr size_t chunk_size = 32;
            size_t next = i - std::min(i - begin, chunk_size);
            auto right_cost = static_cast<Scalar>(0.);
            for (; i > next; --i) {
                right_bbox.extend(bboxes_[prim_ids_[axis][i]]);
                accum_[i] = right_cost = config_.sah.get_leaf_cost(i, end, right_bbox);
            }
            // Every `chunk_size` elements, check that we are not above the maximum cost
            if (right_cost > best_split.cost) {
                first_right = i;
                break;
            }
        }

        // Sweep from the left to the right, computing the full cost
        auto left_bbox = BBox::make_empty();
        for (size_t i = begin; i < first_right; ++i)
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
        for (size_t i = first_right; i < end - 1; ++i) {
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
            auto left_cost = config_.sah.get_leaf_cost(begin, i + 1, left_bbox);
            auto cost = left_cost + accum_[i + 1];
            if (cost < best_split.cost)
                best_split = Split { i + 1, cost, axis };
            else if (left_cost > best_split.cost)
                break;
        }
    }

    BVH_ALWAYS_INLINE void mark_primitives(size_t axis, size_t begin, size_t split_pos, size_t end) {
        for (size_t i = begin; i < split_pos; ++i) marks_[prim_ids_[axis][i]] = true;
        for (size_t i = split_pos; i < end; ++i)   marks_[prim_ids_[axis][i]] = false;
    }

    std::optional<size_t> try_split(const BBox& bbox, size_t begin, size_t end) override {
        // Find the best split over all axes
        auto leaf_cost = config_.sah.get_non_split_cost(begin, end, bbox);
        auto best_split = Split { (begin + end + 1) / 2, leaf_cost, 0 };
        for (size_t axis = 0; axis < Node::dimension; ++axis)
            find_best_split(axis, begin, end, best_split);

        // Make sure that the split is good before proceeding with it
        if (best_split.cost >= leaf_cost) {
            if (end - begin <= config_.max_leaf_size)
                return std::nullopt;

            // If the number of primitives is too high, fallback on a split at the
            // median on the largest axis.
            best_split.pos = (begin + end + 1) / 2;
            best_split.axis = bbox.get_diagonal().get_largest_axis();
        }

        // Partition primitives (keeping the order intact so that the next recursive calls do not
        // need to sort primitives again).
        mark_primitives(best_split.axis, begin, best_split.pos, end);
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            if (axis == best_split.axis)
                continue;
            std::stable_partition(
                prim_ids_[axis].begin() + begin,
                prim_ids_[axis].begin() + end,
                [&] (size_t i) { return marks_[i]; });
        }

        return std::make_optional(best_split.pos);
    }
};

} // namespace bvh::v2

#endif
