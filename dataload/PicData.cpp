
#include "PicData.h"

void PicData::createOneHotVector(int i)
{
    Matrix mat = Matrix(1,10);
    mat[i] = 1.0;
    label = mat;
}

PicData::PicData() : 
    mat(Matrix(28,28)), label(Matrix(1, 10))
{
}
PicData::PicData(std::vector<float> vec, int i) :
    mat(Matrix(28,28, vec))
{
    createOneHotVector(i);
}


