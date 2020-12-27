#pragma once

#include "../mathlib/Matrix.h"


namespace dataload
{
class PicData 
{

    // Matrices
private:
    mathlib::Matrix mat;
    std::vector<int> labels;

    

    // Constructors / destructor
public:
    PicData(std::vector<float> vec, std::vector<int> labels, int rows, int cols);

    // Methods
public:
    const mathlib::Matrix getMat() const noexcept;  
    const std::vector<int> getLabels() const noexcept;
};
}