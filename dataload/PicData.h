#pragma once

#include "../mathlib/Matrix.h"


namespace dataload
{
class PicData 
{

    // Matrices
private:
    mathlib::Matrix mat;
    mathlib::Matrix label;

    void createOneHotVector(int i);

    // Constructors / destructor
public:
    PicData(std::vector<float> vec, int label, int rows, int cols);

    // Methods
public:
    const mathlib::Matrix getMat() const noexcept;  
    const mathlib::Matrix getLabel() const noexcept;
};
}