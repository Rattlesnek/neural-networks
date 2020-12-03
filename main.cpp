#include <iostream>
#include "MathLib.hpp"

using namespace mathlib;

int main()
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
    std::cout << mat1;
    std::cout << mat2;
    std::cout << outcome;
    Matrix Omat(3,3);
    for (int i = 0; i < 9; i++){
        Omat[i] = i;
    }
    Matrix Tmat = Omat.T();
    std::cout << "Original";
    std::cout << Omat;
    std::cout << "TRANSPOSED" << std::endl;
    std::cout << Tmat;

    return 0;
}