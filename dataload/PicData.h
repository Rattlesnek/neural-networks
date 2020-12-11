#pragma once

#include "Matrix.h"

using namespace mathlib;

class PicData 
{
private:
    Matrix mat;
    Matrix label;

    void createOneHotVector(int i);
public:
    PicData();
    PicData(std::vector<float> vec, int i);
};
