#pragma once
#include "Matrix.h"

namespace mathlib
{

class ErrorFunc
{
    // Constructors / destructor
public:
    ErrorFunc() = delete;

    // Methods
public:
    
    static Matrix softMax(Matrix input_copy);

    static Matrix softmaxCrossentropyWithLogits(Matrix input_copy, const std::vector<int>& label);
    
    static Matrix gradSoftmaxCrossentropyWithLogits(const Matrix& input, const std::vector<int>& label);
};

}