#pragma once

#include "Matrix.h"

using namespace mathlib;

class PicData 
{

    // Matrices
private:
    Matrix mat;
    Matrix label;

    void createOneHotVector(int i);

    // Constructors / destructor
public:
    PicData(std::vector<float> vec, int label, int rows, int cols);

    // Methods
public:
    const Matrix getMat() const noexcept;
    const Matrix getLabel() const noexcept;
};
