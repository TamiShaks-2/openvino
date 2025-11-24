#include <gtest/gtest.h>
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp" // מכיל את TensorMap
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include <set>

// הגדרת מרחבי השמות הפנימיים
namespace ofep = ov::frontend::pytorch;
namespace {

    // הצהרת הפונקציה המקורית (מוגדרת בקובץ אחר)
    extern std::shared_ptr<ov::Node> build_multi_dot_chain(const ofep::NodeContext& ctx,
                                                           const std::vector<std::shared_ptr<ov::Node>>& mats2d,
                                                           const std::vector<std::pair<int64_t, int64_t>>& dims,
                                                           bool all_static_2d);

    // פונקציות מדומות (MOCK functions) – חיוניות לקישור
    std::shared_ptr<ov::Node> build_left_chain(const ofep::NodeContext& ctx, 
                                               const std::vector<std::shared_ptr<ov::Node>>& mats2d) {
        if (mats2d.size() < 2) return nullptr;
        return std::make_shared<ov::op::v0::MatMul>(mats2d[0]->output(0), mats2d[1]->output(0), false, false);
    }

    bool mco_dp_from_dims(const std::vector<std::pair<int64_t, int64_t>>& dims, 
                          std::vector<std::vector<int64_t>>& s) {
        if (dims.size() == 4) {
            s.assign(dims.size(), std::vector<int64_t>(dims.size(), 0));
            s[0][3] = 2; 
            s[0][2] = 0; 
            s[1][3] = 1; 
            return true;
        }
        
        if (dims.size() > 2) {
            for (size_t i = 0; i < dims.size() - 1; ++i) {
                if (dims[i].second != dims[i+1].first) {
                    return false; // Shape mismatch
                }
            }
        }
        return false;
    }

    std::shared_ptr<ov::Node> build_by_split(const ofep::NodeContext& ctx,
                                             const std::vector<std::shared_ptr<ov::Node>>& mats2d,
                                             const std::vector<std::vector<int64_t>>& s,
                                             int i, int j) {
        if (i == j) {
            return mats2d[i];
        }
        int k = s[i][j];
        auto left = build_by_split(ctx, mats2d, s, i, k);
        auto right = build_by_split(ctx, mats2d, s, k + 1, j);
        return std::make_shared<ov::op::v0::MatMul>(left->output(0), right->output(0), false, false);
    }
} // namespace anonymous

using namespace testing;
using namespace ov::op; 

class MultiDotOptimizationTest : public Test {
protected:
    // *** משתמשים בפוינטר חכם כדי לאחסן את הקונטקסט ***
    std::shared_ptr<ofep::NodeContext> context_ptr; 

    MultiDotOptimizationTest() {
        // *** התיקון הקריטי: קריאה לקונסטרקטור המלא (6 ארגומנטים) עם ערכי ברירת מחדל בטוחים ***
        context_ptr = std::make_shared<ofep::NodeContext>(
            nullptr,              // 1: TorchDecoder (shared_ptr)
            ofep::TensorMap{},    // 2: TensorMap (empty map)
            nullptr,              // 3: Output map (shared_ptr)
            nullptr,              // 4: Parameters (shared_ptr)
            nullptr,              // 5: Op ports (shared_ptr)
            nullptr               // 6: TranslateSession* (raw pointer)
        );
    }

    std::shared_ptr<ov::op::v0::Parameter> create_param(ov::Shape shape) {
        return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    }
};

// ------------------- N=3 TEST CASES ---------------------

TEST_F(MultiDotOptimizationTest, N3_RightOptimal_Chain) {
    auto A = create_param({10, 100}); 
    auto B = create_param({100, 5});
    auto C = create_param({5, 50});

    std::vector<std::pair<int64_t, int64_t>> dims = {{10, 100}, {100, 5}, {5, 50}};
    std::vector<std::shared_ptr<ov::Node>> mats2d = {A, B, C};
    
    // *** שינוי: מעביר את context_ptr באמצעות דה-רפרנס (*) ***
    auto result = build_multi_dot_chain(*context_ptr, mats2d, dims, true);

    auto outer_matmul = std::dynamic_pointer_cast<v0::MatMul>(result);
    ASSERT_NE(outer_matmul, nullptr);
    ASSERT_EQ(outer_matmul->input_value(0).get_node_shared_ptr(), A); 

    auto inner_matmul = std::dynamic_pointer_cast<v0::MatMul>(outer_matmul->input_value(1).get_node_shared_ptr());
    ASSERT_NE(inner_matmul, nullptr);
    ASSERT_EQ(inner_matmul->input_value(0).get_node_shared_ptr(), B);
    ASSERT_EQ(inner_matmul->input_value(1).get_node_shared_ptr(), C);
}

TEST_F(MultiDotOptimizationTest, N3_LeftOptimal_Chain) {
    auto A = create_param({50, 5}); 
    auto B = create_param({5, 100});
    auto C = create_param({100, 10});

    std::vector<std::pair<int64_t, int64_t>> dims = {{50, 5}, {5, 100}, {100, 10}};
    std::vector<std::shared_ptr<ov::Node>> mats2d = {A, B, C};
    
    // *** שינוי: מעביר את context_ptr באמצעות דה-רפרנס (*) ***
    auto result = build_multi_dot_chain(*context_ptr, mats2d, dims, true);

    auto outer_matmul = std::dynamic_pointer_cast<v0::MatMul>(result);
    ASSERT_NE(outer_matmul, nullptr);
    ASSERT_EQ(outer_matmul->input_value(1).get_node_shared_ptr(), C); 

    auto inner_matmul = std::dynamic_pointer_cast<v0::MatMul>(outer_matmul->input_value(0).get_node_shared_ptr());
    ASSERT_NE(inner_matmul, nullptr);
    ASSERT_EQ(inner_matmul->input_value(0).get_node_shared_ptr(), A);
    ASSERT_EQ(inner_matmul->input_value(1).get_node_shared_ptr(), B);
}

// ------------------- N=4 TEST CASE (DP) ---------------------

TEST_F(MultiDotOptimizationTest, N4_DP_Chain) {
    auto A = create_param({30, 35}); auto B = create_param({35, 15});
    auto C = create_param({15, 5}); auto D = create_param({5, 10});

    std::vector<std::pair<int64_t, int64_t>> dims = {{30, 35}, {35, 15}, {15, 5}, {5, 10}};
    std::vector<std::shared_ptr<ov::Node>> mats2d = {A, B, C, D};

    // *** שינוי: מעביר את context_ptr באמצעות דה-רפרנס (*) ***
    auto result = build_multi_dot_chain(*context_ptr, mats2d, dims, true);

    auto outer_matmul = std::dynamic_pointer_cast<v0::MatMul>(result);
    ASSERT_NE(outer_matmul, nullptr);
    ASSERT_EQ(outer_matmul->input_value(1).get_node_shared_ptr(), D);

    auto inner_left = std::dynamic_pointer_cast<v0::MatMul>(outer_matmul->input_value(0).get_node_shared_ptr());
    ASSERT_NE(inner_left, nullptr);
    ASSERT_EQ(inner_left->input_value(0).get_node_shared_ptr(), A); 

    auto inner_right = std::dynamic_pointer_cast<v0::MatMul>(inner_left->input_value(1).get_node_shared_ptr());
    ASSERT_NE(inner_right, nullptr);
    ASSERT_EQ(inner_right->input_value(0).get_node_shared_ptr(), B); 
    ASSERT_EQ(inner_right->input_value(1).get_node_shared_ptr(), C);
}

// TEST 4: בדיקת נפילה על צורות לא תואמות (Shape Mismatch)
TEST_F(MultiDotOptimizationTest, IncompatibleShapes_FallBack) {
    auto A = create_param({10, 20}); 
    auto B = create_param({30, 40}); 
    auto C = create_param({40, 50}); 

    std::vector<std::pair<int64_t, int64_t>> dims = {{10, 20}, {30, 40}, {40, 50}};
    std::vector<std::shared_ptr<ov::Node>> mats2d = {A, B, C};
    
    // *** שינוי: מעביר את context_ptr באמצעות דה-רפרנס (*) ***
    // מצפה לחריגה/כשל בגלל shape mismatch
    EXPECT_ANY_THROW(build_multi_dot_chain(*context_ptr, mats2d, dims, true));
}