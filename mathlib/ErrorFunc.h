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

    static float categoricalCrossentropy(const Matrix& predictions, const Matrix& labels);

    static float meanSquareError(const Matrix& predictions, const Matrix& labels);
    
    static Matrix SoftMax(const Matrix& input, const Matrix& label);

    static float SoftmaxCrossentropyWithLogits(const Matrix& input, const Matrix& label);
    
    static Matrix GradSoftmaxCrossEntropyWithLogits(const Matrix& input, const Matrix& label);
};

}