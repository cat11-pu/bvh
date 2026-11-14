#ifndef BVH_V2_MINI_TREE_BUILDER_H
#define BVH_V2_MINI_TREE_BUILDER_H

#include "bvh/v2/sweep_sah_builder.h"
#include "bvh/v2/binned_sah_builder.h"
#include "bvh/v2/thread_pool.h"
#include "bvh/v2/executor.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <limits>
#include <cassert>

namespace bvh::v2 {

/// Multi-threaded top-down builder that partitions primitives using a grid. Multiple instances
/// of a single-threaded builder are run in parallel on that partition, generating many small
/// trees. Finally, a top-level tree is built on these smaller trees to form the final BVH.
/// This builder is inspired by
/// "Rapid Bounding Volume Hierarchy Generation using Mini Trees", by P. Ganestam et al.
template <typename Node, typename MortonCode = uint32_t>
class MiniTreeBuilder {
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config : TopDownSahBuilder<Node>::Config {
        /// Flag that turns on/off mini-tree pruning.
        bool enable_pruning = true;

        /// Threshold on the area of a mini-tree node above which it is pruned, expressed in
        /// fraction of the area of bounding box around the entire set of primitives.
        Scalar pruning_area_ratio = static_cast<Scalar>(0.01);

        /// Minimum number of primitives per parallel task.
        size_t parallel_threshold = 1024;

        /// Log of the dimension of the grid used to split the workload horizontally.
        size_t log2_grid_dim = 4;

        /// Number of mini-trees above which the top-level BVH is built in parallel,
        /// by grouping spatially-close mini-trees and building a sub-BVH per group.
        size_t top_build_threshold = 4096;
    };

    /// Starts building a BVH with the given primitive data. The build algorithm is multi-threaded,
    /// and runs on the given thread pool.
    [[nodiscard]] BVH_ALWAYS_INLINE static Bvh<Node> build(
        ThreadPool& thread_pool,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        MiniTreeBuilder builder(thread_pool, bboxes, centers, config);
        auto mini_trees = builder.build_mini_trees();
        auto roots = config.enable_pruning
            ? builder.find_pruned_roots(mini_trees)
            : MiniTreeBuilder::make_default_roots(mini_trees);
        return builder.build_top_bvh(mini_trees, roots);
    }

private:
    friend struct BuildTask;

    struct BuildTask {
        MiniTreeBuilder* builder;
        Bvh<Node>& bvh;
        std::vector<size_t> prim_ids;

        BuildTask(
            MiniTreeBuilder* builder,
            Bvh<Node>& bvh,
            std::vector<size_t>&& prim_ids)
            : builder(builder)
            , bvh(bvh)
            , prim_ids(std::move(prim_ids))
        {}

        BVH_ALWAYS_INLINE void run() {
            // Build directly over the global primitive arrays, avoiding any copy
            // of the bounding boxes and centers for this set of primitives.
            // Note: The counting sort used to bin primitives already guarantees
            // that rebuilds produce the same BVH.
            bvh = BinnedSahBuilder<Node>::build(
                builder->bboxes_, builder->centers_, std::move(prim_ids), builder->config_);
        }
    };

    ParallelExecutor executor_;
    std::span<const BBox> bboxes_;
    std::span<const Vec> centers_;
    const Config& config_;

    BVH_ALWAYS_INLINE MiniTreeBuilder(
        ThreadPool& thread_pool,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : executor_(thread_pool, 1)
        , bboxes_(bboxes)
        , centers_(centers)
        , config_(config)
    {
        assert(bboxes.size() == centers.size());
    }

    std::vector<Bvh<Node>> build_mini_trees() {
        // Compute the bounding box of all centers
        auto center_bbox = executor_.reduce(0, bboxes_.size(), BBox::make_empty(),
            [this] (BBox& bbox, size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                    bbox.extend(centers_[i]);
            },
            [] (BBox& bbox, const BBox& other) { bbox.extend(other); });

        assert(config_.log2_grid_dim <= std::numeric_limits<MortonCode>::digits / Node::dimension);
        auto bin_count = size_t{1} << (config_.log2_grid_dim * Node::dimension);
        auto grid_dim = size_t{1} << config_.log2_grid_dim;
        auto grid_scale = Vec(static_cast<Scalar>(grid_dim)) * safe_inverse(center_bbox.get_diagonal());
        auto grid_offset = -center_bbox.min * grid_scale;

        auto bin_of = [&] (size_t i) {
            auto p = robust_max(fast_mul_add(centers_[i], grid_scale, grid_offset), Vec(0));
            auto x = std::min(grid_dim - 1, static_cast<size_t>(p[0]));
            auto y = std::min(grid_dim - 1, static_cast<size_t>(p[1]));
            auto z = std::min(grid_dim - 1, static_cast<size_t>(p[2]));
            return morton_encode(x, y, z) & (bin_count - 1);
        };

        // Place primitives in bins using a two-pass counting sort. Compared to keeping one
        // vector per bin, this performs a single allocation for all the primitive indices.
        const size_t prim_count = bboxes_.size();
        const size_t thread_count = std::max<size_t>(size_t{1}, executor_.thread_pool.get_thread_count());
        const size_t chunk_count = std::min(thread_count, prim_count);
        const size_t chunk_size = (prim_count + chunk_count - 1) / chunk_count;

        std::vector<std::vector<size_t>> counts(chunk_count, std::vector<size_t>(bin_count, 0));
        executor_.for_each(0, chunk_count, [&] (size_t begin, size_t end) {
            for (size_t c = begin; c < end; ++c) {
                auto first = c * chunk_size, last = std::min(first + chunk_size, prim_count);
                for (size_t i = first; i < last; ++i)
                    counts[c][bin_of(i)]++;
            }
        });

        // Lay bins out consecutively, with each chunk's indices contiguous within a bin
        std::vector<size_t> bin_offsets(bin_count + 1, 0);
        for (size_t b = 0; b < bin_count; ++b)
            for (size_t c = 0; c < chunk_count; ++c)
                bin_offsets[b + 1] += counts[c][b];
        for (size_t b = 0; b < bin_count; ++b)
            bin_offsets[b + 1] += bin_offsets[b];

        std::vector<size_t> sorted_ids(prim_count);
        {
            std::vector<std::vector<size_t>> cursors(chunk_count, std::vector<size_t>(bin_count));
            for (size_t b = 0; b < bin_count; ++b) {
                size_t offset = bin_offsets[b];
                for (size_t c = 0; c < chunk_count; ++c) {
                    cursors[c][b] = offset;
                    offset += counts[c][b];
                }
            }
            executor_.for_each(0, chunk_count, [&] (size_t begin, size_t end) {
                for (size_t c = begin; c < end; ++c) {
                    auto first = c * chunk_size, last = std::min(first + chunk_size, prim_count);
                    auto& cursor = cursors[c];
                    for (size_t i = first; i < last; ++i)
                        sorted_ids[cursor[bin_of(i)]++] = i;
                }
            });
        }

        // Merge small bins and skip empty ones. Since bins are contiguous ranges in
        // `sorted_ids`, merged bins are simply larger contiguous ranges.
        // Note: Merging small bins will deteriorate the quality of the top BVH if there is no
        // pruning, since it will then produce larger mini-trees. For this reason, it is only enabled
        // when mini-tree pruning is enabled.
        std::vector<std::pair<size_t, size_t>> task_ranges;
        for (size_t i = 0; i < bin_count;) {
            size_t j = i + 1;
            if (config_.enable_pruning) {
                size_t size = bin_offsets[j] - bin_offsets[i];
                for (; j < bin_count && size + (bin_offsets[j + 1] - bin_offsets[j]) <= config_.parallel_threshold; ++j)
                    size += bin_offsets[j + 1] - bin_offsets[j];
            }
            if (bin_offsets[j] > bin_offsets[i])
                task_ranges.emplace_back(bin_offsets[i], bin_offsets[j]);
            i = j;
        }

        // Iterate over bins to collect groups of primitives and build BVHs over them in parallel
        std::vector<Bvh<Node>> mini_trees(task_ranges.size());
        ThreadPool::TaskGroup group;
        for (size_t i = 0; i < task_ranges.size(); ++i) {
            auto [begin, end] = task_ranges[i];
            std::vector<size_t> prim_ids(sorted_ids.begin() + begin, sorted_ids.begin() + end);
            auto task = new BuildTask(this, mini_trees[i], std::move(prim_ids));
            executor_.thread_pool.push(group, [task] (size_t) { task->run(); delete task; });
        }
        executor_.thread_pool.wait(group);

        return mini_trees;
    }

    static std::vector<std::pair<size_t, size_t>> make_default_roots(
        const std::vector<Bvh<Node>>& mini_trees)
    {
        std::vector<std::pair<size_t, size_t>> roots(mini_trees.size());
        for (size_t i = 0; i < mini_trees.size(); ++i)
            roots[i] = std::make_pair(i, size_t{0});
        return roots;
    }

    /// Finds the roots of the sub-trees to use as leaves of the top-level BVH, by culling
    /// mini-tree nodes whose area is above a fraction of the average mini-tree root area.
    std::vector<std::pair<size_t, size_t>> find_pruned_roots(const std::vector<Bvh<Node>>& mini_trees) {
        // Compute the area threshold based on the area of the entire set of primitives
        auto avg_area = static_cast<Scalar>(0.);
        for (auto& mini_tree : mini_trees)
            avg_area += mini_tree.get_root().get_bbox().get_half_area();
        avg_area /= static_cast<Scalar>(mini_trees.size());
        auto threshold = avg_area * config_.pruning_area_ratio;

        // Cull nodes whose area is above the threshold, in parallel. The per-chunk
        // results are concatenated in chunk order to keep the build deterministic.
        const size_t thread_count = std::max<size_t>(size_t{1}, executor_.thread_pool.get_thread_count());
        const size_t chunk_count = std::min(thread_count, mini_trees.size());
        const size_t chunk_size = (mini_trees.size() + chunk_count - 1) / chunk_count;
        std::vector<std::vector<std::pair<size_t, size_t>>> per_chunk_roots(chunk_count);
        executor_.for_each(0, chunk_count, [&] (size_t begin, size_t end) {
            for (size_t c = begin; c < end; ++c) {
                auto& out = per_chunk_roots[c];
                std::vector<size_t> stack;
                auto last = std::min((c + 1) * chunk_size, mini_trees.size());
                for (size_t i = c * chunk_size; i < last; ++i) {
                    stack.push_back(0);
                    auto& mini_tree = mini_trees[i];
                    while (!stack.empty()) {
                        auto node_id = stack.back();
                        auto& node = mini_tree.nodes[node_id];
                        stack.pop_back();
                        if (node.get_bbox().get_half_area() < threshold || node.is_leaf()) {
                            out.emplace_back(i, node_id);
                        } else {
                            stack.push_back(node.index.first_id());
                            stack.push_back(node.index.first_id() + 1);
                        }
                    }
                }
            }
        });
        std::vector<std::pair<size_t, size_t>> pruned_roots;
        for (auto& roots : per_chunk_roots)
            pruned_roots.insert(pruned_roots.end(), roots.begin(), roots.end());
        return pruned_roots;
    }

    Bvh<Node> build_top_bvh(
        std::vector<Bvh<Node>>& mini_trees,
        const std::vector<std::pair<size_t, size_t>>& roots)
    {
        // Build a BVH using the given sub-tree roots as leaves
        std::vector<Vec> centers(roots.size());
        std::vector<BBox> bboxes(roots.size());
        executor_.for_each(0, roots.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i] = mini_trees[roots[i].first].nodes[roots[i].second].get_bbox();
                centers[i] = bboxes[i].get_center();
            }
        });

        typename SweepSahBuilder<Node>::Config config = config_;
        config.max_leaf_size = config.min_leaf_size = 1; // Needs to have only one sub-tree in each leaf
        auto bvh = roots.size() > config_.top_build_threshold
            ? SweepSahBuilder<Node>::build_parallel(executor_, bboxes, centers, config,
                4 * std::max<size_t>(size_t{1}, executor_.thread_pool.get_thread_count()))
            : SweepSahBuilder<Node>::build(bboxes, centers, config);

        // Count the nodes and primitives of every sub-tree, in parallel
        std::vector<size_t> node_counts(roots.size());
        std::vector<size_t> prim_counts(roots.size());
        executor_.for_each(0, roots.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto& nodes = mini_trees[roots[i].first].nodes;
                size_t node_count = 0, prim_count = 0;
                std::vector<size_t> stack;
                stack.push_back(roots[i].second);
                while (!stack.empty()) {
                    auto node_id = stack.back();
                    stack.pop_back();
                    auto& node = nodes[node_id];
                    ++node_count;
                    if (node.is_leaf()) {
                        prim_count += node.index.prim_count();
                    } else {
                        stack.push_back(node.index.first_id() + 0);
                        stack.push_back(node.index.first_id() + 1);
                    }
                }
                node_counts[i] = node_count;
                prim_counts[i] = prim_count;
            }
        });

        // Compute the offsets to apply to node and primitive indices. Sub-tree nodes
        // are laid out after the top-level nodes, without their root (which is merged
        // into the leaves of the top-level BVH).
        std::vector<size_t> node_offsets(roots.size());
        std::vector<size_t> prim_offsets(roots.size());
        size_t node_count = bvh.nodes.size();
        size_t prim_count = 0;
        for (size_t i = 0; i < roots.size(); ++i) {
            node_offsets[i] = node_count;
            prim_offsets[i] = prim_count;
            node_count += node_counts[i] - 1; // Skip root node
            prim_count += prim_counts[i];
        }

        // Fix the root of every sub-tree so that it can be merged into the leaves of
        // the top-level BVH, in parallel
        std::vector<Node> fixed_roots(roots.size());
        executor_.for_each(0, roots.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto root = mini_trees[roots[i].first].nodes[roots[i].second];
                root.index.set_first_id(root.is_leaf() ? prim_offsets[i] : node_offsets[i]);
                fixed_roots[i] = root;
            }
        });

        // Make the leaves of the top BVH point to the right sub-tree nodes
        for (auto& node : bvh.nodes) {
            if (!node.is_leaf())
                continue;
            assert(node.index.prim_count() == 1);
            size_t tree_id = bvh.prim_ids[node.index.first_id()];
            node = fixed_roots[tree_id];
        }

        // Copy the sub-tree nodes and primitives into the final arrays, in parallel
        bvh.nodes.resize(node_count);
        bvh.prim_ids.resize(prim_count);
        executor_.for_each(0, roots.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto& src = mini_trees[roots[i].first];
                auto& src_root = src.nodes[roots[i].second];
                size_t next_node = node_offsets[i];
                size_t next_prim = prim_offsets[i];
                if (src_root.is_leaf()) {
                    std::copy_n(
                        src.prim_ids.begin() + src_root.index.first_id(),
                        src_root.index.prim_count(),
                        bvh.prim_ids.begin() + next_prim);
                    continue;
                }
                std::vector<std::pair<size_t, size_t>> stack;
                stack.emplace_back(src_root.index.first_id() + 0, next_node + 0);
                stack.emplace_back(src_root.index.first_id() + 1, next_node + 1);
                next_node += 2;
                while (!stack.empty()) {
                    auto [src_id, dst_id] = stack.back();
                    stack.pop_back();
                    auto& src_node = src.nodes[src_id];
                    Node dst_node = src_node;
                    if (src_node.is_leaf()) {
                        std::copy_n(
                            src.prim_ids.begin() + src_node.index.first_id(),
                            src_node.index.prim_count(),
                            bvh.prim_ids.begin() + next_prim);
                        dst_node.index.set_first_id(next_prim);
                        next_prim += src_node.index.prim_count();
                    } else {
                        dst_node.index.set_first_id(next_node);
                        stack.emplace_back(src_node.index.first_id() + 0, next_node + 0);
                        stack.emplace_back(src_node.index.first_id() + 1, next_node + 1);
                        next_node += 2;
                    }
                    bvh.nodes[dst_id] = dst_node;
                }
            }
        });

        return bvh;
    }

};

} // namespace bvh::v2

#endif
