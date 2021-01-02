#pragma once

#include "Matrix.h"
#include "PicData.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

namespace dataload 
{
class DataLoader
{
    //Files
private:
    std::ifstream ifImages;
    std::ifstream ifLabels;

    //Constructors / destructor
public:
    DataLoader(std::string pathVec, std::string pathLabels);
    DataLoader(std::string pathVec);
    ~DataLoader();

    // Methods
public:
    std::vector<float> loadPicture(int rows, int cols);
    std::vector<int> loadLabel();
    std::vector<PicData> loadAllData(int rows, int cols);
    std::vector<mathlib::Matrix> loadAllPictures(int rows, int cols);
    std::vector<PicData> loadNOfEach(int n, int rows, int cols);
    
};

}