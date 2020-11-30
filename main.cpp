#include <iostream>
#include "MathLib.hpp"

using namespace mathlib;

int main()
{   
    // Example
    Matrix mat{3, 3};

    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(2, 0) = 3;

    mat.print();

    return 0;
}