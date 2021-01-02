

#include "PreprocessingUtils.h"
using namespace dataload;
std::tuple<std::vector<PicData>, std::vector<PicData>> PreprocessingUtils::splitDataValidTrain(float percentageValid, std::vector<PicData> data)
{
    std::vector<PicData> valid;
    std::vector<PicData> train;
    int limitEachClass = ((int) (float)data.size() * percentageValid) / 10;
    std::cout << limitEachClass << "<- limit" << std::endl;
    int count = 0;
    std::vector<int> counts(10);
    std::cout << "startcounts: "<<  std::endl;
    
    for (auto i = counts.begin(); i != counts.end(); ++i)
    {
    std::cout << *i << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < data.size(); i++)
    {
        auto pic = data[i];
        if (counts[pic.getLabels()[0]] < limitEachClass)
        {
            counts[pic.getLabels()[0]] += 1;
            valid.emplace_back(pic);
        }
        else
        {
            train.emplace_back(pic);
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