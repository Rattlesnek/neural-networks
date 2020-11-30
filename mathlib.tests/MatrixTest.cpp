#include <gtest/gtest.h>
#include "MathLib.hpp"

using namespace mathlib;

TEST(MatrixTest, MatrixAssignment)
{
    Matrix mat(5, 5);
    mat(0, 0) = 10.0;

    EXPECT_EQ(mat(0, 0), 10.0);
}
