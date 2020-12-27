#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "Matrix.h"
#include "PicData.h"

namespace dataload
{

class PreprocessingUtils
{
    // Constructors / destructor
public:
    PreprocessingUtils() = delete;

    // Methods
public:
    // percentageValid : number between 0.f - 1.f
    static std::tuple<std::vector<PicData>, std::vector<PicData>> splitDataValidTrain(float percentageValid, std::vector<PicData> data);
};
}