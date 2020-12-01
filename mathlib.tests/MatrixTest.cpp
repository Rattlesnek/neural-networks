#include <gtest/gtest.h>
#include "MathLib.hpp"

using namespace mathlib;

TEST(MatrixTest, MatrixAssignment)
{
    Matrix mat(5, 5);
    mat(0, 0) = 10.0;

    EXPECT_EQ(mat(0, 0), 10.0);
}
TEST(MatrixTest, MatrixAddition)
{
    Matrix mat1(2,2);
    Matrix mat2(2,2);
    mat2(0,0) = 1;
    mat1(0,0) = 2;
    mat1 + mat2;
    EXPECT_EQ(mat1(0,0),3);
}
TEST(MatrixTest, MatrixMultiplication)
{
    Matrix mat1(2,3);
    Matrix mat2(3,2);
    mat1(0,0) = 1;
    mat1(0,1) = 2;
    mat1(0,2) = 3;
    mat1(1,0) = 4;
    mat1(1,1) = 5;
    mat1(1,2) = 6;
    mat2(0,0) = 7;
    mat2(0,1) = 8;
    mat2(1,0) = 9;
    mat2(1,1) = 10;
    mat2(2,0) = 11;
    mat2(2,1) = 12;
    Matrix outcome = mat1 * mat2;
    EXPECT_EQ(outcome(0,0), 58);
    EXPECT_EQ(outcome(0,1), 64);
    EXPECT_EQ(outcome(1,0), 139);
    EXPECT_EQ(outcome(1,1), 154);
}
