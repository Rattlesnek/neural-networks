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

std::vector<float> DataLoader::loadPicture(int rows, int cols)
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
    while (std::getline(ssline, val, ','))
    {
        vec.emplace_back(std::stof(val));
    }
    

    return vec;
}

std::vector<int> DataLoader::loadLabel()
{
    std::string label;
    std::vector<int> oneHotIndex(1);
    if (std::getline(labels, label))
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
    std::vector<PicData> pics;
    int count = 0;
    std::vector<int> counts(10);
    
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
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
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    return pics;
}

std::tuple<std::vector<PicData>, std::vector<PicData>> DataLoader::getValidTrain(int rows, int cols)
{
    std::vector<PicData> valid;
    std::vector<PicData> train;
    int count = 0;
    std::vector<int> counts(10);
    
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    while (true)
    {
        
        try
        {
            auto pic = PicData(loadPicture(rows, cols), loadLabel(), rows, cols);
            if (counts[pic.getLabels()[0]] < 600)
            {
                valid.emplace_back(pic);
                counts[pic.getLabels()[0]] += 1;
            }
            else
            {
                train.emplace_back(pic);
            }
        }
        catch(const EOFException& e)
        {
            
            break;
        }
        
    }
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    std::cout << "Valid data size: " << valid.size() << std::endl;
    std::cout << "Train data size: " << train.size() << std::endl;

    std::tuple<std::vector<PicData>, std::vector<PicData>> output(valid, train);
    
    return output;
}


