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
TEST(ErrorFuncTest, SoftMaxCrossEntropy)
{
    Matrix testIn(1,3,{4, 3, 1});
            Matrix testL(1,3,{1,0,0});
            std::cout << "print the GSCEWL matrix here pythonOutput:([-0.29461549  0.25949646  0.03511903]):" << std::endl;
            std::cout << ErrorFunc::GradSoftmaxCrossEntropyWithLogits(testIn, testL);
            std::cout << "print the SoftmaxCrossentropyWithLogits here pythonOutput:(0.34901222):" << std::endl;
            std::cout << ErrorFunc::SoftmaxCrossentropyWithLogits(testIn, testL) << std::endl;
}