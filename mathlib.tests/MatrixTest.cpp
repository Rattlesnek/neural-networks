#include <gtest/gtest.h>
#include "MathLib.hpp"

using namespace mathlib;

TEST(MatrixTest, MatrixAssignment)
{
    Matrix mat(5, 5);
    mat(3, 4) = 10.f;

    EXPECT_EQ(mat(3, 4), 10.f);
    EXPECT_EQ(mat(4, 3), 0.f);
}

TEST(MatrixTest, MatrixAddition)
{
    Matrix mat1(2, 2, {1, 2, 3, 4});
    Matrix mat2(2, 2, {9, 8, 7, 6});

    auto mat = mat1 + mat2;
    EXPECT_EQ(mat(0, 0), 10.f);
    EXPECT_EQ(mat(0, 1), 10.f);
    EXPECT_EQ(mat(1, 0), 10.f);
    EXPECT_EQ(mat(1, 1), 10.f);
}

TEST(MatrixTest, MatrixAdditionException)
{
    Matrix mat1(2,3);
    Matrix mat2(3,2);

    EXPECT_THROW(mat1 + mat2, MatrixException);
}

TEST(MatrixTest, MatrixMultiplication)
{
    Matrix mat1(2,3,{1,2,3,4,5,6});
    Matrix mat2(3,2,{7,8,9,10,11,12});

    Matrix outcome = mat1 * mat2;
    EXPECT_EQ(outcome(0,0), 58.f);
    EXPECT_EQ(outcome(0,1), 64.f);
    EXPECT_EQ(outcome(1,0), 139.f);
    EXPECT_EQ(outcome(1,1), 154.f);
}

TEST(MatrixTest, TransposeBasic)
{
    Matrix mat(3, 3);
    for (int i = 0; i < 9; i++){
        mat[i] = i;
    }

    Matrix Tmat = mat.T();
    EXPECT_EQ(Tmat(1,1), 4.f);
    EXPECT_EQ(Tmat(0,1), 3.f);
    EXPECT_EQ(Tmat(2,1), 5.f);
}

TEST(MatrixTest, ApplyFunctionBasic)
{
    Matrix m(2,2);

    m.applyFunc([](float x) -> float { return x + 3.8f; });
    EXPECT_EQ(m(0,0), 3.8f);
    EXPECT_EQ(m(1,0), 3.8f);
}

TEST(MatrixTest, BasicSumTest)
{
    Matrix m(2, 2, {1.0,1.0,1.0,1.0});
    Matrix m1(2, 2, {0.9,1.1,0.54,0.46});
    
    EXPECT_EQ(m.sum(), 4.f);
    EXPECT_EQ(m1.sum(), 3.f);
}
