// // src/frontends/pytorch/src/op/translate_multi_dot.cpp
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
// #include "openvino/src/frontends/pytorch/src/utils.hpp"

// using namespace ov;
// using ov::frontend::pytorch::get_list_as_outputs;

// namespace ov {
// namespace frontend {
// namespace pytorch {
// namespace op {

// // בניית שרשרת כפל לפי טבלת הפיצול של MCO
// static std::shared_ptr<Node> build_chain(
//         const std::vector<std::shared_ptr<Node>>& mats,
//         const std::vector<std::vector<int64_t>>& split) {
//     std::function<std::shared_ptr<Node>(int,int)> rec =
//         [&](int i, int j) -> std::shared_ptr<Node> {
//             if (i == j) return mats[(size_t)i];
//             int k = static_cast<int>(split[(size_t)i][(size_t)j]);
//             auto L = rec(i, k);
//             auto R = rec(k + 1, j);
//             return std::make_shared<ov::op::v0::MatMul>(L, R, false, false);
//         };
//     return rec(0, static_cast<int>(mats.size()) - 1);
// }

// OutputVector translate_multi_dot(const NodeContext& ctx) {
//     // aten::linalg_multi_dot(Tensor[] tensors, Tensor? out=None) -> Tensor
//     FRONT_END_OP_CONVERSION_CHECK(ctx.get_input_size() >= 1,
//         "linalg_multi_dot expects a list/tuple of tensors as input 0");

//     // 1) שליפת הרשימה מהקלט הראשון של האופרטור
//     // OutputVector ins = get_list_as_outputs(ctx, 0);
//     const ov::Output<ov::Node> list0 = ctx.get_input(0);
//     auto deq = get_list_as_outputs(list0, /*unsqueeze_for_concat=*/false);
//     OutputVector ins(deq.begin(), deq.end());
    
//     FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
//         "linalg_multi_dot expects at least 2 tensors");

//     // 2) טיפול בקצוות 1D → 2D זמנית
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

//     // אמצעיים חייבים להיות 2D
//     for (size_t i = 1; i + 1 < ins.size(); ++i) {
//         FRONT_END_OP_CONVERSION_CHECK(
//             is_rank2(ins[i].get_partial_shape()) ||
//             ins[i].get_partial_shape().rank().compatible(2),
//             "linalg_multi_dot: middle tensors must be 2D");
//     }

//     // 3) בחירת סדר כפל
//     // אפשר להריץ MCO רק אם כל הטנזורים 2D עם ממדים סטטיים
//     auto all_static_2d = [&]{
//         for (const auto& t : ins) {
//             const auto ps = t.get_partial_shape();
//             if (!ps.rank().is_static() || ps.rank().get_length() != 2) return false;
//             if (!ps[0].is_static() || !ps[1].is_static()) return false;
//         }
//         return true;
//     }();

//     // ממירים ל-Node* לצורך MatMul
//     std::vector<std::shared_ptr<Node>> mats;
//     mats.reserve(ins.size());
//     for (auto& t : ins) mats.push_back(t.get_node_shared_ptr());

//     std::shared_ptr<Node> prod;

//     if (ins.size() == 2) {
//         prod = std::make_shared<ov::op::v0::MatMul>(mats[0], mats[1], false, false);
//     } else if (ins.size() == 3 && all_static_2d) {
//         // מקרה 3 מטריצות לפי העלות של PyTorch
//         const int64_t a = ins[0].get_partial_shape()[0].get_length();
//         const int64_t b = ins[1].get_partial_shape()[0].get_length();
//         const int64_t c = ins[2].get_partial_shape()[0].get_length();
//         const int64_t d = ins[2].get_partial_shape()[1].get_length();
//         const int64_t cost1 = (a * c) * (b + d); // A@(B@C)
//         const int64_t cost2 = (b * d) * (a + c); // (A@B)@C
//         if (cost1 > cost2) {
//             auto bc = std::make_shared<ov::op::v0::MatMul>(mats[1], mats[2], false, false);
//             prod = std::make_shared<ov::op::v0::MatMul>(mats[0], bc, false, false);
//         } else {
//             auto ab = std::make_shared<ov::op::v0::MatMul>(mats[0], mats[1], false, false);
//             prod = std::make_shared<ov::op::v0::MatMul>(ab, mats[2], false, false);
//         }
//     } else if (ins.size() >= 4 && all_static_2d) {
//         // MCO מלא
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
//         // ממדים דינמיים או דרגות לא סטטיות → שמאלי בטוח
//         prod = mats[0];
//         for (size_t i = 1; i < mats.size(); ++i)
//             prod = std::make_shared<ov::op::v0::MatMul>(prod, mats[i], false, false);
//     }

//     // 4) שחזור צורה: הסרת ה-unsqueeze בקצוות
//     Output<Node> out = prod;
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
// src/frontends/pytorch/src/op/translate_multi_dot.cpp

#include <vector>
#include <memory>
#include <functional>
#include <limits>
#include <deque>

#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
// Correct header for PyTorch FE helpers (provides get_list_as_outputs and checks)
#include "../utils.hpp"


using namespace ov;
using ov::frontend::pytorch::get_list_as_outputs;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// Build MatMul chain from split table produced by Matrix Chain Order (MCO)
static std::shared_ptr<Node> build_chain(
        const std::vector<std::shared_ptr<Node>>& mats,
        const std::vector<std::vector<int64_t>>& split) {
    std::function<std::shared_ptr<Node>(int,int)> rec =
        [&](int i, int j) -> std::shared_ptr<Node> {
            if (i == j) return mats[(size_t)i];
            int k = static_cast<int>(split[(size_t)i][(size_t)j]);
            auto L = rec(i, k);
            auto R = rec(k + 1, j);
            // MatMul expects ov::Output<Node>, so pass output(0)
            return std::make_shared<ov::op::v0::MatMul>(
                L->output(0), R->output(0), /*transpose_a=*/false, /*transpose_b=*/false);
        };
    return rec(0, static_cast<int>(mats.size()) - 1);
}

ov::OutputVector translate_multi_dot(const ov::frontend::pytorch::NodeContext& ctx) {
    // aten::linalg_multi_dot(Tensor[] tensors, Tensor? out=None) -> Tensor
    FRONT_END_OP_CONVERSION_CHECK(ctx.get_input_size() >= 1,
        "linalg_multi_dot expects a list/tuple of tensors as input 0");

    // 1) Extract the list from input 0 using the FE helper you have:
    // Your utils.hpp declares: deque<Output<Node>> get_list_as_outputs(Output<Node>, bool)
    const ov::Output<ov::Node> list0 = ctx.get_input(0);
    auto deq = get_list_as_outputs(list0, /*unsqueeze_for_concat=*/false);
    OutputVector ins(deq.begin(), deq.end());

    FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
        "linalg_multi_dot expects at least 2 tensors");

    // 2) Handle 1D edges → temporarily treat as 2D with Unsqueeze
    const auto axis0  = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    const auto axism1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});

    auto is_rank1 = [](const PartialShape& ps) -> bool {
        return ps.rank().is_static() && ps.rank().get_length() == 1;
    };
    auto is_rank2 = [](const PartialShape& ps) -> bool {
        return ps.rank().is_static() && ps.rank().get_length() == 2;
    };

    const bool first1D = is_rank1(ins.front().get_partial_shape());
    const bool last1D  = is_rank1(ins.back().get_partial_shape());

    if (first1D) {
        ins.front() = std::make_shared<ov::op::v0::Unsqueeze>(ins.front(), axis0);
    } else {
        FRONT_END_OP_CONVERSION_CHECK(
            is_rank2(ins.front().get_partial_shape()) ||
            ins.front().get_partial_shape().rank().compatible(2),
            "linalg_multi_dot: first tensor must be 1D or 2D");
    }

    if (last1D) {
        ins.back() = std::make_shared<ov::op::v0::Unsqueeze>(ins.back(), axism1);
    } else {
        FRONT_END_OP_CONVERSION_CHECK(
            is_rank2(ins.back().get_partial_shape()) ||
            ins.back().get_partial_shape().rank().compatible(2),
            "linalg_multi_dot: last tensor must be 1D or 2D");
    }

    // Middle tensors must be 2D
    for (size_t i = 1; i + 1 < ins.size(); ++i) {
        FRONT_END_OP_CONVERSION_CHECK(
            is_rank2(ins[i].get_partial_shape()) ||
            ins[i].get_partial_shape().rank().compatible(2),
            "linalg_multi_dot: middle tensors must be 2D");
    }

    // 3) Choose multiplication order
    // Run MCO only if all tensors are 2D with static dims
    auto all_static_2d = [&]{
        for (const auto& t : ins) {
            const auto ps = t.get_partial_shape();
            if (!ps.rank().is_static() || ps.rank().get_length() != 2) return false;
            if (!ps[0].is_static() || !ps[1].is_static()) return false;
        }
        return true;
    }();

    // Convert to nodes for MatMul chaining
    std::vector<std::shared_ptr<Node>> mats;
    mats.reserve(ins.size());
    for (auto& t : ins) mats.push_back(t.get_node_shared_ptr());

    std::shared_ptr<Node> prod;

    if (ins.size() == 2) {
        prod = std::make_shared<ov::op::v0::MatMul>(
            mats[0]->output(0), mats[1]->output(0), false, false);
    } else if (ins.size() == 3 && all_static_2d) {
        // 3-matrix case: use PyTorch's cost heuristic
        const int64_t a = ins[0].get_partial_shape()[0].get_length();
        const int64_t b = ins[1].get_partial_shape()[0].get_length();
        const int64_t c = ins[2].get_partial_shape()[0].get_length();
        const int64_t d = ins[2].get_partial_shape()[1].get_length();
        const int64_t cost1 = (a * c) * (b + d); // A@(B@C)
        const int64_t cost2 = (b * d) * (a + c); // (A@B)@C
        if (cost1 > cost2) {
            auto bc = std::make_shared<ov::op::v0::MatMul>(
                mats[1]->output(0), mats[2]->output(0), false, false);
            prod = std::make_shared<ov::op::v0::MatMul>(
                mats[0]->output(0), bc->output(0), false, false);
        } else {
            auto ab = std::make_shared<ov::op::v0::MatMul>(
                mats[0]->output(0), mats[1]->output(0), false, false);
            prod = std::make_shared<ov::op::v0::MatMul>(
                ab->output(0), mats[2]->output(0), false, false);
        }
    } else if (ins.size() >= 4 && all_static_2d) {
        // Full dynamic-programming MCO
        const size_t n = ins.size();
        std::vector<int64_t> p; p.reserve(n + 1);
        p.push_back(ins[0].get_partial_shape()[0].get_length());
        for (size_t i = 0; i < n; ++i)
            p.push_back(ins[i].get_partial_shape()[1].get_length());

        std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));
        std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n, 0));
        for (size_t L = 2; L <= n; ++L) {
            for (size_t i = 0; i + L - 1 < n; ++i) {
                const size_t j = i + L - 1;
                m[i][j] = std::numeric_limits<int64_t>::max();
                for (size_t k = i; k < j; ++k) {
                    const int64_t cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1];
                    if (cost < m[i][j]) {
                        m[i][j] = cost;
                        s[i][j] = static_cast<int64_t>(k);
                    }
                }
            }
        }
        prod = build_chain(mats, s);
    } else {
        // Fallback for dynamic ranks/dims: safe left-associative chain
        prod = mats[0];
        for (size_t i = 1; i < mats.size(); ++i) {
            prod = std::make_shared<ov::op::v0::MatMul>(
                prod->output(0), mats[i]->output(0), false, false);
        }
    }

    // 4) Restore original shape: remove Unsqueeze at edges
    ov::Output<ov::Node> out = prod->output(0);
    if (last1D)
        out = std::make_shared<ov::op::v0::Squeeze>(out, axism1);
    if (first1D)
        out = std::make_shared<ov::op::v0::Squeeze>(out, axis0);

    return { out };
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov
