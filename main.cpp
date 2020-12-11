#include <iostream>
#include "MathLib.hpp"
#include <memory>

using namespace mathlib;
using namespace mathlib::activation;

int main()
{   
    std::unique_ptr<IActivation> activation = std::make_unique<Sigmoid>();
    std::cout << activation->call(Matrix(2, 2, {-1, 0, 1, 2})) << std::endl; 

    Matrix mat(2, 2, {1, 2, 3, 4});

    for (auto item : mat)
    {
        std::cout << item << " ";
    }

    return 0;
}