#pragma once
#include <vector>
#include <memory>
#include <optional>
#include "ILayer.h"
#include "PicData.h"

namespace nnlib
{

class Network
{
    // Fields
private: 
    // added some private attributes to help me with methods
    std::vector<std::shared_ptr<ILayer>> layers;
    float alpha;
    int totalSteps;
    int epoch;
    int epochLimit;
    // Constructors / destructor
public:
    Network(std::vector<std::shared_ptr<ILayer>> layers, float alpha);

    // Methods
public:
    void train(const int numOfEpochs, const int batchSize, const std::vector<dataload::PicData>& trainData, std::vector<dataload::PicData>& validationData);

    std::tuple<float, float> trainOnBatch(const std::vector<dataload::PicData>& batch, int datasetIndex);

    void predict(std::vector<dataload::PicData> validationData);
private:
    bool correctPrediction(const mathlib::Matrix& pred, const std::vector<int>& labels);

};

}
