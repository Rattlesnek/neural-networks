#include <gtest/gtest.h>
#include "MathLib.hpp"

using namespace mathlib;

TEST(ActivationFuncTest, Softmax)
{
    std::vector<double> vec = {8, 5, 0};
    auto result = ActivationFunc::softmax(vec);

    EXPECT_NEAR(result[0], 0.9523, 0.0001);
    EXPECT_NEAR(result[1], 0.0474, 0.0001);
    EXPECT_NEAR(result[2], 0.0003, 0.0001);
}
