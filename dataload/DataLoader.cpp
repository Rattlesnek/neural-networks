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
    int count = 0;
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
        count++;
    }
    std::cout << "number of pics: " << count << std::endl;
    return pics;
}

std::vector<PicData> DataLoader::loadNOfEach(int n, int rows, int cols)
{
    std::vector<PicData> pics;
    int count = 0;
    std::vector<int> counts(10);
    std::cout << "startcounts: "<<  std::endl;
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    while (pics.size() < 10 * n)
    {
        
        try
        {
            auto pic = loadPicture(rows, cols);
            if (counts[pic.getIndex()] < n)
            {
                pics.emplace_back(pic);
                counts[pic.getIndex()] += 1;
            }
        }
        catch(const EOFException& e)
        {
            std::cout << "end of file!\n";
            break;
        }
        
    }
    std::cout << "endcounts: "<< std::endl;
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
    std::cout << "startcounts: "<<  std::endl;
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    while (true)
    {
        
        try
        {
            auto pic = loadPicture(rows, cols);
            if (counts[pic.getIndex()] < 600)
            {
                valid.emplace_back(pic);
                counts[pic.getIndex()] += 1;
            }
            else
            {
                train.emplace_back(pic);
            }
        }
        catch(const EOFException& e)
        {
            std::cout << "end of file!\n";
            break;
        }
        
    }
    std::cout << "endcounts: "<< std::endl;
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


