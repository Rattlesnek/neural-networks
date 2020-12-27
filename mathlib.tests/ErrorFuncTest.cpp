#include <gtest/gtest.h>
#include "MathLib.hpp"
#include <iostream>

using namespace mathlib;
using namespace mathlib::activation;

TEST(ErrorFuncTest, SoftMaxCrossEntropy)
{
    Matrix testIn(1,3,{4, 3, 1});
            std::vector<int> testL ={0};
            std::cout << "print the GSCEWL matrix here pythonOutput:([-0.29461549  0.25949646  0.03511903]):" << std::endl;
            std::cout << ErrorFunc::gradSoftmaxCrossentropyWithLogits(testIn, testL);
            std::cout << "print the SoftmaxCrossentropyWithLogits here pythonOutput:(0.34901222):" << std::endl;
            std::cout << ErrorFunc::softmaxCrossentropyWithLogits(testIn, testL) << std::endl;
}