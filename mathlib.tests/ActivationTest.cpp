#include <gtest/gtest.h>
#include "MathLib.hpp"
#include <memory>
#include <iostream>

using namespace mathlib;
using namespace mathlib::activation;

TEST(ActivationTest, Softmax)
{
    Matrix mat(1, 3, {8, 5, 0});
    std::unique_ptr<IActivation> activation = std::make_unique<Softmax>();

    auto result = activation->call(mat);

    EXPECT_NEAR(result(0, 0), 0.9523, 0.0001);
    EXPECT_NEAR(result(0, 1), 0.0474, 0.0001);
    EXPECT_NEAR(result(0, 2), 0.0003, 0.0001);
}

TEST(ActivationTest, Sigmoid)
{
    Matrix mat(1, 3, {-1000, 0, 1000});
    std::unique_ptr<IActivation> activation = std::make_unique<Sigmoid>();

    auto result = activation->call(mat);

    EXPECT_NEAR(result(0, 0), 0.0, 0.0001);
    EXPECT_NEAR(result(0, 1), 0.5, 0.0001);
    EXPECT_NEAR(result(0, 2), 1.0, 0.0001);
}

TEST(ActivationTest, ReLU)
{
    Matrix mat(2, 2, {-1, 0, 1, 2});
    std::unique_ptr<IActivation> activation = std::make_unique<ReLU>();

    auto result = activation->call(mat);

    EXPECT_NEAR(result(0, 0), 0.0, 0.0001);
    EXPECT_NEAR(result(0, 1), 0.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 1.0, 0.0001);
    EXPECT_NEAR(result(1, 1), 2.0, 0.0001);
}