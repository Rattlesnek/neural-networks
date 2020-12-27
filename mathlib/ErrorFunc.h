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
    
    static Matrix softMax(const Matrix& input);

    static Matrix softmaxCrossentropyWithLogits(const Matrix& input, const std::vector<int>& label);
    
    static Matrix gradSoftmaxCrossentropyWithLogits(const Matrix& input, const std::vector<int>& label);
};

}