
#include "PicData.h"

void PicData::createOneHotVector(int i)
{
    Matrix mat = Matrix(1,10);
    mat[i] = 1.0;
    label = mat;
}

PicData::PicData(std::vector<float> vec, int label, int rows, int cols) :
    mat(Matrix(rows, cols, vec))
{
    mat.applyFunc([](float x) -> float { return x/255; });
    createOneHotVector(label);
}



const Matrix PicData::getMat() const noexcept
{
    return mat;
}

const Matrix PicData::getLabel() const noexcept
{
    return label;
}


