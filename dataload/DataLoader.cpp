#include "DataLoader.h" 


using std::vector;
using std::string;
using std::getline;

PicData DataLoader::loadPicture(std::ifstream& imageFile, std::ifstream& labels)
{
    PicData pic;
    string line;
    string label;
    vector<float> vec;
    std::getline(imageFile, line);
    
    std::stringstream sline(line);
    string val;
    int oneHotIndex;
    while (std::getline(sline, val, ','))
    {
        vec.push_back(std::stof(val));
    }
    if (std::getline(labels, label))
    {
        oneHotIndex = std::stoi(label);
    }
    pic = PicData(vec,oneHotIndex);
    return pic;
}
