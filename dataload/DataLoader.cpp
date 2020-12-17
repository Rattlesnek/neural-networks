#include "DataLoader.h" 
#include "EOFException.h"
#include "DataLoadInternalException.h"

using namespace dataload;

DataLoader::DataLoader(std::string pathVec, std::string pathLabels)
{
    imageVecs.open(pathVec, std::ios::in);
    if(!imageVecs)
    {
        throw DataLoadInternalException("Open file pathVec failed!");
    }
    labels.open(pathLabels, std::ios::in);
    if(!labels)
    {
        throw DataLoadInternalException("Open file pathLabels failed!");
    }
}

DataLoader::~DataLoader()
{
    imageVecs.close();
    labels.close();
}

PicData DataLoader::loadPicture(int rows, int cols)
{
    std::string line;
    std::string label;
    std::vector<float> vec;
    if (!std::getline(imageVecs, line))
    {
        throw EOFException("end of file vectors reached!");
    }
    
    std::stringstream ssline(line);
    std::string val;
    int oneHotIndex;
    while (std::getline(ssline, val, ','))
    {
        vec.emplace_back(std::stof(val));
    }
    if (std::getline(labels, label))
    {
        oneHotIndex = std::stoi(label);
    }
    else
    {
        throw EOFException("end of file labels reached!");
    }
    PicData pic(vec, oneHotIndex, rows, cols);

    return pic;
}

std::vector<PicData> DataLoader::loadAllData(int rows, int cols)
{
    std::vector<PicData> pics;
    while (true)
    {
        try
        {
            pics.emplace_back(loadPicture(rows, cols));
        }
        catch(const std::exception& e)
        {
            std::cout << "end of file!\n";
            break;
        }
    }
    return pics;
}
