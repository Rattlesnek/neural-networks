#pragma once
#include <vector>
#include <memory>
#include "ILayer.h"
#include "PicData.h"

namespace nnlib
{

class Network
{
    // Fields
private:
    std::vector<std::shared_ptr<ILayer>> layers;

    // Constructors / destructor
public:
    Network(std::vector<std::shared_ptr<ILayer>> layers);

    // Methods
public:
    void train(const int numOfEpochs, const int batchSize, const std::vector<dataload::PicData>& trainData);

    std::tuple<float, float> trainOnBatch(const std::vector<dataload::PicData>& batch);

    // void predict();

private:
    bool correctPrediction(const mathlib::Matrix& pred, const std::vector<int>& labels);

};

}
