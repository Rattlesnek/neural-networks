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

    static double categoricalCrossentropy(const Matrix& predictions, const Matrix& labels);

};

}