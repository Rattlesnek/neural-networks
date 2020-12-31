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

    // Constructors / destructor
public:
    Network(std::vector<std::shared_ptr<ILayer>> layers);

    // Methods
public:
    void train(const int numOfEpochs, 
        const int batchSize, 
        const float learningRate, 
        const float momentumFactor
, 
        const std::vector<dataload::PicData>& trainData, 
        std::optional<std::vector<dataload::PicData>> validationData = std::nullopt);

    std::tuple<float, float> trainOnBatch(const std::vector<dataload::PicData>& batch, const float alpha, const float momentumFactor);

    std::tuple<float, float> predict(std::vector<dataload::PicData> validationData);

private:
    bool correctPrediction(const mathlib::Matrix& pred, const std::vector<int>& labels);

};

}
