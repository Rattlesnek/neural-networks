#pragma once

#include "Matrix.h"
#include "PicData.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>



class DataLoader
{
private:
    std::ifstream imageVecs;
    std::ifstream labels;
public:
    DataLoader(std::string pathVec, std::string pathLabels);

public:
    PicData loadPicture(std::ifstream& image, std::ifstream& labels);

    void normPicData(std::vector<float>& picVec);
};