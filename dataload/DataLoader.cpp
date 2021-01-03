#include "DataLoader.h" 
#include "EOFException.h"
#include "DataLoadInternalException.h"

using namespace dataload;

DataLoader::DataLoader(std::string pathVec, std::string pathLabels)
{
    ifImages.open(pathVec, std::ios::in);
    if(!ifImages)
    {
        throw DataLoadInternalException("Open file pathVec failed!");
    }
    ifLabels.open(pathLabels, std::ios::in);
    if(!ifLabels)
    {
        throw DataLoadInternalException("Open file pathLabels failed!");
    }
}

DataLoader::DataLoader(std::string pathVec)
{
    ifImages.open(pathVec, std::ios::in);
    if(!ifImages)
    {
        throw DataLoadInternalException("Open file pathVec failed!");
    }
}

DataLoader::~DataLoader()
{
    ifImages.close();

    if (ifLabels.is_open())
    {
        ifLabels.close();
    }   
}

std::vector<float> DataLoader::loadPicture(int rows, int cols)
{
    std::string line;
    std::string label;
    std::vector<float> vec;
    if (!std::getline(ifImages, line))
    {
        throw EOFException("end of file vectors reached!");
    }
    
    std::stringstream ssline(line);
    std::string val;
    while (std::getline(ssline, val, ','))
    {
        vec.emplace_back(std::stof(val));
    }

    return vec;
}

std::vector<int> DataLoader::loadLabel()
{
    if (!ifLabels.is_open())
    {
        throw DataLoadInternalException("Labels stream is not open");
    }
    
    std::string label;
    std::vector<int> oneHotIndex(1);
    if (std::getline(ifLabels, label))
    {
        oneHotIndex[0] = std::stoi(label);
    }
    else
    {
        throw EOFException("end of file labels reached!");
    }
    return oneHotIndex;
}

std::vector<PicData> DataLoader::loadAllData(int rows, int cols)
{
    if (!ifLabels.is_open())
    {
        throw DataLoadInternalException("Labels stream is not open");
    }

    std::vector<PicData> pics;
    while (true)
    {
        try
        {
            PicData pic = PicData(loadPicture(rows, cols), loadLabel(), rows, cols);
            pics.emplace_back(pic);
        }
        catch(const std::exception& e)
        {
            break;
        }
    }
    
    return pics;
}

std::vector<mathlib::Matrix> DataLoader::loadAllPictures(int rows, int cols)
{
    std::vector<mathlib::Matrix> pics;
    while (true)
    {
        try
        {
            auto pic = mathlib::Matrix(rows,cols,loadPicture(rows, cols));
            pic.applyFunc([](float x) -> float { return x / 255.f; });
            pics.emplace_back(pic);
        }
        catch(const std::exception& e)
        {
            break;
        }
        
    }
    
    return pics;
}

std::vector<PicData> DataLoader::loadNOfEach(int n, int rows, int cols)
{
    if (!ifLabels.is_open())
    {
        throw DataLoadInternalException("Labels stream is not open");
    }
    
    std::vector<PicData> pics;
    int count = 0;
    std::vector<int> counts(10);
    
    while (pics.size() < 10 * n)
    {
        try
        {
            auto pic = PicData(loadPicture(rows, cols), loadLabel(), rows, cols);
            if (counts[pic.getLabels()[0]] < n)
            {
                pics.emplace_back(pic);
                counts[pic.getLabels()[0]] += 1;
            }
        }
        catch(const EOFException& e)
        {
            
            break;
        }
    }
    
    return pics;
}
