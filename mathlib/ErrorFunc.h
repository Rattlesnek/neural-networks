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

};

}