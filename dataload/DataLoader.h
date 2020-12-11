#pragma once

#include "Matrix.h"
#include "PicData.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>



class DataLoader
{
    //Files
private:
    std::ifstream imageVecs;
    std::ifstream labels;

    //Constructors / destructor
public:
    DataLoader(std::string pathVec, std::string pathLabels);
    ~DataLoader();

    // Methods
public:
    PicData loadPicture();
    std::vector<PicData> loadAllData();
};