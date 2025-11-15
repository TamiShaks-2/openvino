// #include <vector>
// #include <memory>
// #include <functional>
// #include <limits>
// #include <deque>

// #include "openvino/op/constant.hpp"
// #include "openvino/op/matmul.hpp"
// #include "openvino/op/squeeze.hpp"
// #include "openvino/op/unsqueeze.hpp"
// #include "openvino/frontend/pytorch/node_context.hpp"
// // Correct header for PyTorch FE helpers (provides get_list_as_outputs and checks)
// #include "../utils.hpp"


// using namespace ov;
// using ov::frontend::pytorch::get_list_as_outputs;

// namespace ov {
// namespace frontend {
// namespace pytorch {
// namespace op {

// // Build MatMul chain from split table produced by Matrix Chain Order (MCO)
// static std::shared_ptr<Node> build_chain(
//         const std::vector<std::shared_ptr<Node>>& mats,
//         const std::vector<std::vector<int64_t>>& split) {
//     std::function<std::shared_ptr<Node>(int,int)> rec =
//         [&](int i, int j) -> std::shared_ptr<Node> {
//             if (i == j) return mats[(size_t)i];
//             int k = static_cast<int>(split[(size_t)i][(size_t)j]);
//             auto L = rec(i, k);
//             auto R = rec(k + 1, j);
//             // MatMul expects ov::Output<Node>, so pass output(0)
//             return std::make_shared<ov::op::v0::MatMul>(
//                 L->output(0), R->output(0), /*transpose_a=*/false, /*transpose_b=*/false);
//         };
//     return rec(0, static_cast<int>(mats.size()) - 1);
// }

// ov::OutputVector translate_multi_dot(const ov::frontend::pytorch::NodeContext& ctx) {
//     // aten::linalg_multi_dot(Tensor[] tensors, Tensor? out=None) -> Tensor
//     FRONT_END_OP_CONVERSION_CHECK(ctx.get_input_size() >= 1,
//         "linalg_multi_dot expects a list/tuple of tensors as input 0");

//     // 1) Extract the list from input 0 using the FE helper you have:
//     // Your utils.hpp declares: deque<Output<Node>> get_list_as_outputs(Output<Node>, bool)
//     const ov::Output<ov::Node> list0 = ctx.get_input(0);
//     auto deq = get_list_as_outputs(list0, /*unsqueeze_for_concat=*/false);
//     OutputVector ins(deq.begin(), deq.end());

//     FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
//         "linalg_multi_dot expects at least 2 tensors");

//     // 2) Handle 1D edges → temporarily treat as 2D with Unsqueeze
//     const auto axis0  = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
//     const auto axism1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});

//     auto is_rank1 = [](const PartialShape& ps) -> bool {
//         return ps.rank().is_static() && ps.rank().get_length() == 1;
//     };
//     auto is_rank2 = [](const PartialShape& ps) -> bool {
//         return ps.rank().is_static() && ps.rank().get_length() == 2;
//     };

//     const bool first1D = is_rank1(ins.front().get_partial_shape());
//     const bool last1D  = is_rank1(ins.back().get_partial_shape());

//     if (first1D) {
//         ins.front() = std::make_shared<ov::op::v0::Unsqueeze>(ins.front(), axis0);
//     } else {
//         FRONT_END_OP_CONVERSION_CHECK(
//             is_rank2(ins.front().get_partial_shape()) ||
//             ins.front().get_partial_shape().rank().compatible(2),
//             "linalg_multi_dot: first tensor must be 1D or 2D");
//     }

//     if (last1D) {
//         ins.back() = std::make_shared<ov::op::v0::Unsqueeze>(ins.back(), axism1);
//     } else {
//         FRONT_END_OP_CONVERSION_CHECK(
//             is_rank2(ins.back().get_partial_shape()) ||
//             ins.back().get_partial_shape().rank().compatible(2),
//             "linalg_multi_dot: last tensor must be 1D or 2D");
//     }

//     // Middle tensors must be 2D
//     for (size_t i = 1; i + 1 < ins.size(); ++i) {
//         FRONT_END_OP_CONVERSION_CHECK(
//             is_rank2(ins[i].get_partial_shape()) ||
//             ins[i].get_partial_shape().rank().compatible(2),
//             "linalg_multi_dot: middle tensors must be 2D");
//     }

//     // 3) Choose multiplication order
//     // Run MCO only if all tensors are 2D with static dims
//     auto all_static_2d = [&]{
//         for (const auto& t : ins) {
//             const auto ps = t.get_partial_shape();
//             if (!ps.rank().is_static() || ps.rank().get_length() != 2) return false;
//             if (!ps[0].is_static() || !ps[1].is_static()) return false;
//         }
//         return true;
//     }();

//     // Convert to nodes for MatMul chaining
//     std::vector<std::shared_ptr<Node>> mats;
//     mats.reserve(ins.size());
//     for (auto& t : ins) mats.push_back(t.get_node_shared_ptr());

//     std::shared_ptr<Node> prod;

//     if (ins.size() == 2) {
//         prod = std::make_shared<ov::op::v0::MatMul>(
//             mats[0]->output(0), mats[1]->output(0), false, false);
//     } else if (ins.size() == 3 && all_static_2d) {
//         // 3-matrix case: use PyTorch's cost heuristic
//         const int64_t a = ins[0].get_partial_shape()[0].get_length();
//         const int64_t b = ins[1].get_partial_shape()[0].get_length();
//         const int64_t c = ins[2].get_partial_shape()[0].get_length();
//         const int64_t d = ins[2].get_partial_shape()[1].get_length();
//         const int64_t cost1 = (a * c) * (b + d); // A@(B@C)
//         const int64_t cost2 = (b * d) * (a + c); // (A@B)@C
//         if (cost1 > cost2) {
//             auto bc = std::make_shared<ov::op::v0::MatMul>(
//                 mats[1]->output(0), mats[2]->output(0), false, false);
//             prod = std::make_shared<ov::op::v0::MatMul>(
//                 mats[0]->output(0), bc->output(0), false, false);
//         } else {
//             auto ab = std::make_shared<ov::op::v0::MatMul>(
//                 mats[0]->output(0), mats[1]->output(0), false, false);
//             prod = std::make_shared<ov::op::v0::MatMul>(
//                 ab->output(0), mats[2]->output(0), false, false);
//         }
//     } else if (ins.size() >= 4 && all_static_2d) {
//         // Full dynamic-programming MCO
//         const size_t n = ins.size();
//         std::vector<int64_t> p; p.reserve(n + 1);
//         p.push_back(ins[0].get_partial_shape()[0].get_length());
//         for (size_t i = 0; i < n; ++i)
//             p.push_back(ins[i].get_partial_shape()[1].get_length());

//         std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));
//         std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n, 0));
//         for (size_t L = 2; L <= n; ++L) {
//             for (size_t i = 0; i + L - 1 < n; ++i) {
//                 const size_t j = i + L - 1;
//                 m[i][j] = std::numeric_limits<int64_t>::max();
//                 for (size_t k = i; k < j; ++k) {
//                     const int64_t cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1];
//                     if (cost < m[i][j]) {
//                         m[i][j] = cost;
//                         s[i][j] = static_cast<int64_t>(k);
//                     }
//                 }
//             }
//         }
//         prod = build_chain(mats, s);
//     } else {
//         // Fallback for dynamic ranks/dims: safe left-associative chain
//         prod = mats[0];
//         for (size_t i = 1; i < mats.size(); ++i) {
//             prod = std::make_shared<ov::op::v0::MatMul>(
//                 prod->output(0), mats[i]->output(0), false, false);
//         }
//     }

//     // 4) Restore original shape: remove Unsqueeze at edges
//     ov::Output<ov::Node> out = prod->output(0);
//     if (last1D)
//         out = std::make_shared<ov::op::v0::Squeeze>(out, axism1);
//     if (first1D)
//         out = std::make_shared<ov::op::v0::Squeeze>(out, axis0);

//     return { out };
// }

// } // namespace op
// } // namespace pytorch
// } // namespace frontend
// } // namespace ov



// #include <vector>
// #include <memory>
// #include <functional>
// #include <limits>
// #include <deque>

// #include "openvino/op/constant.hpp"
// #include "openvino/op/matmul.hpp"
// #include "openvino/op/squeeze.hpp"
// #include "openvino/op/unsqueeze.hpp"
// #include "openvino/frontend/pytorch/node_context.hpp" 
// #include "../utils.hpp"
// #include <iostream>
// // --- BEGIN DEBUG ---
// #pragma message(">>>🎉✨ COMPILING translate_multi_dot.cpp <<<")

// using namespace ov;
// namespace ofep = ov::frontend::pytorch;
// #include <fstream>
// #include <cstdlib>
// #include <iostream>


// namespace ov {
// namespace frontend {
// namespace pytorch {
// namespace op {
// //FRONT_END_OP_CONVERSION_CHECK(false, "### translate_multi_dot CALLED ###");

// //------------------------------------------------------------------------------
// // Helper: construct a node and register it via mark_node; always return Node ptr
// //------------------------------------------------------------------------------
// template <class T, class... Args>
// static std::shared_ptr<ov::Node> mk(const ofep::NodeContext& ctx, Args&&... args) {
//     auto n = std::make_shared<T>(std::forward<Args>(args)...);
//     return ctx.mark_node(n); // returns std::shared_ptr<ov::Node>
// }

// //------------------------------------------------------------------------------
// // Build a MatMul chain from an MCO split table (k indices).
// //------------------------------------------------------------------------------
// static std::shared_ptr<Node> build_chain(const std::vector<std::shared_ptr<Node>>& mats,
//                                          const std::vector<std::vector<int64_t>>& split,
//                                          const ofep::NodeContext& ctx) {
//     std::function<std::shared_ptr<Node>(int,int)> rec =
//         [&](int i, int j) -> std::shared_ptr<Node> {
//             if (i == j) return mats[(size_t)i];
//             int k = static_cast<int>(split[(size_t)i][(size_t)j]);
//             auto L = rec(i, k);
//             auto R = rec(k + 1, j);
//             return mk<ov::op::v0::MatMul>(ctx, L->output(0), R->output(0), false, false);
//         };
//     return rec(0, static_cast<int>(mats.size()) - 1);
// }

// static inline void ptfe_dbg() {
//     if (const char* dbg = std::getenv("OV_FE_PT_DEBUG"); dbg && std::string(dbg)=="1") {
//         std::cerr << "[PT-FE] translate_multi_dot CALLED\n";
//         std::cerr.flush();
//     }
// }
// //------------------------------------------------------------------------------
// // Translator for aten::linalg_multi_dot
// //
// // Steps:
// //  1) Extract input list/tuple of tensors using FE helper.
// //  2) Treat 1-D endpoints as (1,N) / (N,1) via Unsqueeze.
// //  3) Choose multiplication order:
// //       - 2 tensors: single MatMul.
// //       - 3 tensors & static 2D: PyTorch heuristic.
// //       - 4+ tensors & static 2D: full DP (Matrix Chain Order).
// //       - otherwise: left-associative fallback.
// //  4) Squeeze back to restore 1-D endpoint ranks.
// //------------------------------------------------------------------------------
// ov::OutputVector translate_multi_dot(const ofep::NodeContext& ctx) {
//     throw std::runtime_error("🔥 DEBUG: translate_multi_dot TRIGGERED!");
//     const char* dbg = std::getenv("OV_FE_PT_DEBUG");
//     ptfe_dbg();

//     FRONT_END_OP_CONVERSION_CHECK(ctx.get_input_size() >= 1,
//         "linalg_multi_dot expects a list/tuple of tensors as input 0");
//     // Extract list elements into OutputVector
//     const ov::Output<ov::Node> list0 = ctx.get_input(0);
//     auto deq = ofep::get_list_as_outputs(list0, /*unsqueeze_for_concat=*/false);
//     OutputVector ins(deq.begin(), deq.end());

//     FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
//         "linalg_multi_dot expects at least 2 tensors");

//     // Axis constants (keep the Node pointer and use output(0) explicitly)
//     auto axis0_node  = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{0});
//     auto axism1_node = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{-1});
//     auto axis0  = axis0_node->output(0);
//     auto axism1 = axism1_node->output(0);

//     // Rank helpers
//     auto rank_is = [](const PartialShape& ps, int64_t r) {
//         return ps.rank().is_static() ? (ps.rank().get_length() == r) : ps.rank().compatible(r);
//     };

//     const bool first1D = rank_is(ins.front().get_partial_shape(), 1);
//     const bool last1D  = rank_is(ins.back().get_partial_shape(), 1);

//     // First tensor: if 1D → (1,N); else must be 2D
//     if (first1D) {
//         ins.front() = mk<ov::op::v0::Unsqueeze>(ctx, ins.front(), axis0)->output(0);
//     } else {
//         FRONT_END_OP_CONVERSION_CHECK(rank_is(ins.front().get_partial_shape(), 2),
//             "linalg_multi_dot: first tensor must be 1D or 2D");
//     }

//     // Last tensor: if 1D → (N,1); else must be 2D
//     if (last1D) {
//         ins.back() = mk<ov::op::v0::Unsqueeze>(ctx, ins.back(), axism1)->output(0);
//     } else {
//         FRONT_END_OP_CONVERSION_CHECK(rank_is(ins.back().get_partial_shape(), 2),
//             "linalg_multi_dot: last tensor must be 1D or 2D");
//     }

//     // Middle tensors must be 2D
//     for (size_t i = 1; i + 1 < ins.size(); ++i) {
//         FRONT_END_OP_CONVERSION_CHECK(rank_is(ins[i].get_partial_shape(), 2),
//             "linalg_multi_dot: middle tensors must be 2D");
//     }

//     // Convert to nodes for chain building
//     std::vector<std::shared_ptr<Node>> mats;
//     mats.reserve(ins.size());
//     for (auto& t : ins) mats.push_back(t.get_node_shared_ptr());

//     std::shared_ptr<Node> prod;

//     // All-static 2D check enables heuristic/DP
//     auto all_static_2d = [&]{
//         for (const auto& t : ins) {
//             const auto ps = t.get_partial_shape();
//             if (!ps.rank().is_static() || ps.rank().get_length() != 2) return false;
//             if (!ps[0].is_static() || !ps[1].is_static()) return false;
//         }
//         return true;
//     }();

//     if (ins.size() == 2) {
//         // Two tensors → single MatMul
//         prod = mk<ov::op::v0::MatMul>(ctx, mats[0]->output(0), mats[1]->output(0), false, false);
//     } else if (ins.size() == 3 && all_static_2d) {
//         // Three tensors → PyTorch’s cost heuristic
//         const int64_t a = ins[0].get_partial_shape()[0].get_length();
//         const int64_t b = ins[1].get_partial_shape()[0].get_length();
//         const int64_t c = ins[2].get_partial_shape()[0].get_length();
//         const int64_t d = ins[2].get_partial_shape()[1].get_length();
//         const int64_t cost1 = (a * c) * (b + d); // A @ (B @ C)
//         const int64_t cost2 = (b * d) * (a + c); // (A @ B) @ C
//         if (cost1 > cost2) {
//             auto bc = mk<ov::op::v0::MatMul>(ctx, mats[1]->output(0), mats[2]->output(0), false, false);
//             prod = mk<ov::op::v0::MatMul>(ctx, mats[0]->output(0), bc->output(0), false, false);
//         } else {
//             auto ab = mk<ov::op::v0::MatMul>(ctx, mats[0]->output(0), mats[1]->output(0), false, false);
//             prod = mk<ov::op::v0::MatMul>(ctx, ab->output(0), mats[2]->output(0), false, false);
//         }
//     } else if (ins.size() >= 4 && all_static_2d) {
//         // Four or more tensors → full DP (Matrix Chain Order)
//         const size_t n = ins.size();
//         std::vector<int64_t> p; p.reserve(n + 1);
//         p.push_back(ins[0].get_partial_shape()[0].get_length());
//         for (size_t i = 0; i < n; ++i) p.push_back(ins[i].get_partial_shape()[1].get_length());

//         std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));
//         std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n, 0));
//         for (size_t L = 2; L <= n; ++L) {
//             for (size_t i = 0; i + L - 1 < n; ++i) {
//                 const size_t j = i + L - 1;
//                 m[i][j] = std::numeric_limits<int64_t>::max();
//                 for (size_t k = i; k < j; ++k) {
//                     const int64_t cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1];
//                     if (cost < m[i][j]) { m[i][j] = cost; s[i][j] = static_cast<int64_t>(k); }
//                 }
//             }
//         }
//         prod = build_chain(mats, s, ctx);
//     } else {
//         // General/dynamic fallback: left-associative chain
//         auto cur = mats[0];
//         for (size_t i = 1; i < mats.size(); ++i) {
//             cur = mk<ov::op::v0::MatMul>(ctx, cur->output(0), mats[i]->output(0), false, false);
//         }
//         prod = cur;
//     }

//     // Restore original ranks on endpoints
//     ov::Output<ov::Node> out = prod->output(0);
//     if (last1D)  out = mk<ov::op::v0::Squeeze>(ctx, out, axism1)->output(0);
//     if (first1D) out = mk<ov::op::v0::Squeeze>(ctx, out, axis0)->output(0);
    
//     return { out };
// }

// } // namespace op
// } // namespace pytorch
// } // namespace frontend
// } // namespace ov
// translate_multi_dot.cpp
// translate_multi_dot.cpp
#include <vector>
#include <memory>
#include <limits>
#include <cstdint>
#include <utility>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
 // --- BEGIN DEBUG ---
#pragma message(">>>🎪✡️ COMPILING translate_multi_dot.cpp <<<")

// Try to include PT-FE utils if available
#if __has_include("openvino/frontend/pytorch/utils.hpp")
  #include "openvino/frontend/pytorch/utils.hpp"
  #define OV_PTFE_HAVE_UTILS 1
#else
  #define OV_PTFE_HAVE_UTILS 0
#endif

using namespace ov;
namespace ofep = ov::frontend::pytorch;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// ----------------------------- helpers --------------------------------------
// ENGLISH: helper to create & mark nodes
template <class T, class... Args>
static std::shared_ptr<Node> mk(const ofep::NodeContext& ctx, Args&&... args) {
    auto n = std::make_shared<T>(std::forward<Args>(args)...);
    return ctx.mark_node(n);
}

// ENGLISH: rank or -1 if dynamic
static int64_t rank_of(const Output<Node>& x) {
    auto ps = x.get_partial_shape();
    if (!ps.rank().is_static()) return -1;
    return ps.rank().get_length();
}

// ENGLISH: static 2D shape, or empty if rank/dims dynamic
static std::vector<int64_t> static_shape_2d(const Output<Node>& x) {
    std::vector<int64_t> s;
    auto ps = x.get_partial_shape();
    if (!ps.rank().is_static() || ps.rank().get_length() != 2) return s;
    for (int i = 0; i < 2; ++i) {
        if (!ps[i].is_static()) return {};
        s.push_back(static_cast<int64_t>(ps[i].get_length()));
    }
    return s;
}

// ENGLISH: unsqueeze 1D vector to 2D row/col
static std::shared_ptr<Node> unsqueeze_if_1d(const ofep::NodeContext& ctx,
                                             const Output<Node>& x, int64_t axis) {
    auto r = rank_of(x);
    if (r == 1) {
        auto ax = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{axis});
        return mk<ov::op::v0::Unsqueeze>(ctx, x, ax);
    }
    return x.get_node_shared_ptr();
}

// ENGLISH: squeeze back to 1D if exactly one of endpoints was 1D
static std::shared_ptr<Node> maybe_squeeze_back_to_1d(const ofep::NodeContext& ctx,
                                                      const Output<Node>& x,
                                                      bool first_was_1d,
                                                      bool last_was_1d) {
    if (!(first_was_1d ^ last_was_1d)) return x.get_node_shared_ptr();
    auto ps = x.get_partial_shape();
    if (ps.rank().is_static() && ps.rank().get_length() == 2) {
        int64_t axis = first_was_1d ? 0 : 1;
        auto ax = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{axis});
        return mk<ov::op::v0::Squeeze>(ctx, x, ax);
    }
    return x.get_node_shared_ptr();
}

// ENGLISH: try to extract list elements if input is a list; return true if succeeded
static bool try_dequeue_list(const ofep::NodeContext& ctx,
                             const Output<Node>& in,
                             std::vector<Output<Node>>& out) {
#if OV_PTFE_HAVE_UTILS
    // Variant 1: function in root pytorch namespace
    try {
        auto deq = ov::frontend::pytorch::get_list_as_outputs(in, /*unsqueeze_for_concat=*/false);
        if (!deq.empty()) {
            out.insert(out.end(), deq.begin(), deq.end());
            return true;
        }
    } catch (...) { /* fallthrough */ }
    // Variant 2: function under ::utils::
    try {
        using ov::frontend::pytorch::utils::get_list_as_outputs;
        auto deq = get_list_as_outputs(in, /*unsqueeze_for_concat=*/false);
        if (!deq.empty()) {
            out.insert(out.end(), deq.begin(), deq.end());
            return true;
        }
    } catch (...) { /* fallthrough */ }
#endif
    return false;
}

// ENGLISH: collect inputs – (A) single-list port or (B) positional ports
static std::vector<Output<Node>> collect_inputs(const ofep::NodeContext& ctx) {
    std::vector<Output<Node>> xs;
    if (ctx.get_input_size() == 1) {
        const auto in0 = ctx.get_input(0);
        if (try_dequeue_list(ctx, in0, xs)) {
            return xs; // list unpacked successfully
        }
        // treat as a single tensor if not a list / utils not available
        xs.push_back(in0);
        return xs;
    }
    for (size_t i = 0; i < ctx.get_input_size(); ++i)
        xs.push_back(ctx.get_input(i));
    return xs;
}

// ENGLISH: DP for matrix-chain order. dims[i] = (rows_i, cols_i). Returns split table s.
static bool mco_dp_from_dims(const std::vector<std::pair<int64_t,int64_t>>& dims,
                             std::vector<std::vector<int64_t>>& s_out) {
    const int n = (int)dims.size();
    if (n < 2) return false;

    // Build p[] of length n+1 where dims[i] = (p[i], p[i+1])
    std::vector<int64_t> p(n+1);
    p[0] = dims[0].first;
    for (int i = 0; i < n; ++i) {
        if (dims[i].first <= 0 || dims[i].second <= 0) return false;
        if (i > 0 && dims[i-1].second != dims[i].first) return false; // inner mismatch
        p[i+1] = dims[i].second;
    }

    std::vector<std::vector<long double>> m(n, std::vector<long double>(n, 0));
    std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n, -1));

    for (int L = 2; L <= n; ++L) {
        for (int i = 0; i <= n - L; ++i) {
            int j = i + L - 1;
            m[i][j] = std::numeric_limits<long double>::infinity();
            for (int k = i; k < j; ++k) {
                long double cost = m[i][k] + m[k+1][j] + (long double)p[i]*p[k+1]*p[j+1];
                if (cost < m[i][j]) {
                    m[i][j] = cost;
                    s[i][j] = k;
                }
            }
        }
    }
    s_out = std::move(s);
    return true;
}

// ENGLISH: recursively build according to split table
static std::shared_ptr<Node> build_by_split(const ofep::NodeContext& ctx,
                                            const std::vector<std::shared_ptr<Node>>& mats2d,
                                            const std::vector<std::vector<int64_t>>& s,
                                            int i, int j) {
    if (i == j) return mats2d[i];
    int k = s[i][j];
    auto left  = build_by_split(ctx, mats2d, s, i, k);
    auto right = build_by_split(ctx, mats2d, s, k+1, j);
    return mk<ov::op::v0::MatMul>(ctx, left, right, false, false);
}

// ENGLISH: left-associative fallback (safe for dynamic shapes)
static std::shared_ptr<Node> build_left_chain(const ofep::NodeContext& ctx,
                                              const std::vector<std::shared_ptr<Node>>& mats2d) {
    auto acc = mats2d[0];
    for (size_t i = 1; i < mats2d.size(); ++i) {
        acc = mk<ov::op::v0::MatMul>(ctx, acc->output(0), mats2d[i]->output(0), false, false);
    }
    return acc;
}

// ENGLISH: decide matrix multiplication order and build the chain
static std::shared_ptr<Node> build_multi_dot_chain(const ofep::NodeContext& ctx,
                                                   const std::vector<std::shared_ptr<Node>>& mats2d,
                                                   const std::vector<std::pair<int64_t,int64_t>>& dims,
                                                   bool all_static_2d) {
    const size_t n = mats2d.size();
    FRONT_END_OP_CONVERSION_CHECK(n >= 2,
        "linalg_multi_dot requires at least 2 tensors; got ", n);

    // Case 1: exactly two tensors -> single MatMul (no need for shape heuristics)
    if (n == 2) {
        return mk<ov::op::v0::MatMul>(ctx,
                                      mats2d[0]->output(0),
                                      mats2d[1]->output(0),
                                      false,
                                      false);
    }

    // If shapes are not fully static 2D, use safe left-associative chain
    if (!all_static_2d) {
        return build_left_chain(ctx, mats2d);
    }

    // At this point: all_static_2d == true, dims.size() == n

    // Case 2: exactly three tensors -> PyTorch-like heuristic
    if (n == 3) {
        // Assume A=(a,b), B=(b,c), C=(c,d)
        const int64_t a = dims[0].first;   // rows of A
        const int64_t b = dims[0].second;  // cols of A == rows of B
        const int64_t c = dims[1].second;  // cols of B == rows of C
        const int64_t d = dims[2].second;  // cols of C

        const long long cost_right = (long long)a * c * (b + d); // A @ (B @ C)
        const long long cost_left  = (long long)b * d * (a + c); // (A @ B) @ C

        if (cost_right > cost_left) {
            auto ab = mk<ov::op::v0::MatMul>(ctx,
                                             mats2d[0]->output(0),
                                             mats2d[1]->output(0),
                                             false,
                                             false);
            return mk<ov::op::v0::MatMul>(ctx,
                                          ab->output(0),
                                          mats2d[2]->output(0),
                                          false,
                                          false);
        } else {
            auto bc = mk<ov::op::v0::MatMul>(ctx,
                                             mats2d[1]->output(0),
                                             mats2d[2]->output(0),
                                             false,
                                             false);
            return mk<ov::op::v0::MatMul>(ctx,
                                          mats2d[0]->output(0),
                                          bc->output(0),
                                          false,
                                          false);
        }
    }

    // Case 3: four or more tensors with static 2D shapes -> full DP (Matrix-Chain Order)
    std::vector<std::vector<int64_t>> s;
    bool ok = mco_dp_from_dims(dims, s);
    FRONT_END_OP_CONVERSION_CHECK(ok, "linalg_multi_dot: incompatible static shapes");
    return build_by_split(ctx, mats2d, s, /*i=*/0, /*j=*/static_cast<int>(n) - 1);
}

ov::OutputVector translate_multi_dot(const ofep::NodeContext& ctx) {
    // 1) Gather inputs
    auto ins = collect_inputs(ctx);
    FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
        "linalg_multi_dot requires at least 2 tensors; got ", ins.size());

    // 2) Align dtypes (convert all to first's dtype if needed)
    auto et0 = ins[0].get_element_type();
    for (size_t i = 1; i < ins.size(); ++i) {
        if (ins[i].get_element_type() != et0) {
            ins[i] = mk<ov::op::v0::Convert>(ctx, ins[i], et0);
        }
    }

    // 3) Normalize endpoints 1D->2D (row / col), middles must be 2D or dynamic rank
    const bool first_was_1d = (rank_of(ins.front()) == 1);
    const bool last_was_1d  = (rank_of(ins.back())  == 1);

    std::vector<std::shared_ptr<Node>> mats;
    mats.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        if (i == 0) {
            mats.push_back(unsqueeze_if_1d(ctx, ins[i], /*axis=*/0));
        } else if (i == ins.size() - 1) {
            mats.push_back(unsqueeze_if_1d(ctx, ins[i], /*axis=*/1));
        } else {
            FRONT_END_OP_CONVERSION_CHECK(rank_of(ins[i]) == 2 || rank_of(ins[i]) == -1,
                "linalg_multi_dot expects 2D middle tensors");
            mats.push_back(ins[i].get_node_shared_ptr());
        }
    }

    // 4) Check if all shapes are static 2D and collect dims
    bool all_static_2d = true;
    std::vector<std::pair<int64_t,int64_t>> dims;
    dims.reserve(mats.size());
    for (auto& m : mats) {
        auto s = static_shape_2d(m->output(0));
        if (s.empty()) {
            all_static_2d = false;
            break;
        }
        dims.emplace_back(s[0], s[1]);
    }

    // 5) Delegate chain-building strategy to helper
    std::shared_ptr<Node> prod =
        build_multi_dot_chain(ctx, mats, dims, all_static_2d);

    // 6) squeeze back to 1D if needed (exactly one endpoint was 1D)
    prod = maybe_squeeze_back_to_1d(ctx, prod->output(0), first_was_1d, last_was_1d);

    return { prod };
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov
