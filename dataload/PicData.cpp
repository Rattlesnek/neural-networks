
#include "PicData.h"
#include <stdexcept>


using namespace dataload;
using namespace mathlib;
void PicData::createOneHotVector(int i)
{
    if (i < 0 || i > 9)
    {
        throw std::runtime_error("Wrong label number!");
    }
    label.setDimensions(1,10);
    label[i] = 1.0;
}

PicData::PicData(std::vector<float> vec, int label, int rows, int cols) :
    mat(rows, cols, vec), index(label)
{
    mat.applyFunc([](float x) -> float { return x/255.f; });
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
const int PicData::getIndex() const noexcept
{
    return index;
}


