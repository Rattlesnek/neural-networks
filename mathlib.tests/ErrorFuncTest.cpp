#include <gtest/gtest.h>
#include "MathLib.hpp"
#include <iostream>

using namespace mathlib;
using namespace mathlib::activation;

TEST(ErrorFuncTest, CategoricalCrossentropy)
{
    Matrix labels(1, 3, {1, 0, 0});
    Matrix predictions(1, 3, {0.5, 0.3, 0.2});

    double loss = ErrorFunc::categoricalCrossentropy(predictions, labels);
    EXPECT_NEAR(loss, 0.69314, 0.00001);
}