#include "DataLoader.h" 
#include "EOFException.h"
#include "DataLoadInternalException.h"

using std::vector;
using std::string;
using std::getline;
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

PicData DataLoader::loadPicture()
{
    string line;
    string label;
    vector<float> vec;
    if (!std::getline(imageVecs, line))
    {
        throw EOFException("end of file vectors reached!");
    }
    
    std::stringstream ssline(line);
    string val;
    int oneHotIndex;
    while (std::getline(ssline, val, ','))
    {
        vec.push_back(std::stof(val));
    }
    if (std::getline(labels, label))
    {
        oneHotIndex = std::stoi(label);
    }
    else
    {
        throw EOFException("end of file labels reached!");
    }
    PicData pic(vec, oneHotIndex);

    return pic;
}

std::vector<PicData> DataLoader::loadAllData()
{
    std::vector<PicData> pics;
    while (true)
    {
        try
        {
            pics.push_back(loadPicture());
        }
        catch(const std::exception& e)
        {
            std::cout << "end of file!\n";
            break;
        }
    }
    std::cout << pics[0].getMat() << '\n';
    std::cout << pics[0].getLabel() << '\n';
    return pics;
}
