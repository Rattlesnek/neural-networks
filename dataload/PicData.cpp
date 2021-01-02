
#include "PicData.h"
#include <stdexcept>


using namespace dataload;
using namespace mathlib;

PicData::PicData(std::vector<float> vec, std::vector<int> labels, int rows, int cols) :
    mat(rows, cols, vec), labels(labels)
{
    mat.applyFunc([](float x) -> float { return x / 255.f; });
}

const Matrix PicData::getMat() const noexcept
{
    return mat;
}

const std::vector<int> PicData::getLabels() const noexcept
{
    return labels;
}

