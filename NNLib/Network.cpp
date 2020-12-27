#include "Network.h"
#include <memory>
#include <chrono>
#include <cfloat>

#include "MathLib.hpp"
#include "NetworkException.h"

using namespace nnlib;
using namespace mathlib;
using namespace dataload;

Network::Network(std::vector<std::shared_ptr<ILayer>> layers) :
    layers(std::move(layers))
{

}

void Network::train(const int numOfEpochs, const int batchSize, const std::vector<PicData>& trainData)
{
    if (trainData.size() % batchSize != 0)
    {
        throw NetworkException("TrainData size is not divisible by batchSize.");
    }
    int numOfBatches = trainData.size() / batchSize;
    
    std::cout << "=====================================\n";
    std::cout << "Training\n";
    
    auto totalStartTime = std::chrono::steady_clock::now();
    
    int epoch = 0;
    while (epoch < numOfEpochs)
    {
        std::cout << "------------------------------------------\n";
        std::cout << "Epoch no." << ++epoch << std::endl;
        auto epochStartTime = std::chrono::steady_clock::now();

        float totalBatchAccuracy = 0.f;
        float totalMeanBatchLoss = 0.f;

        // Training
        int datasetIndex = 0;
        while(datasetIndex < (int)trainData.size())
        {           
            std::vector<PicData> batch(trainData.begin() + datasetIndex, trainData.begin() + datasetIndex + batchSize);

            auto [batchAccuracy, meanBatchLoss] = trainOnBatch(batch);
            totalBatchAccuracy += batchAccuracy;
            totalMeanBatchLoss += meanBatchLoss;
            datasetIndex += batchSize;
        }

        auto epochEndTime = std::chrono::steady_clock::now();
        std::cout << std::endl << "Elapsed time in training Epoch no. " << epoch << " : "
            << std::chrono::duration_cast<std::chrono::seconds>(epochEndTime - epochStartTime).count() << " sec" << std::endl;

        float epochAccuracy = totalBatchAccuracy / (float)numOfBatches;
        std::cout << "Epoch accuracy: " << epochAccuracy << std::endl;

        float meanEpochLoss = totalMeanBatchLoss / (float)numOfBatches;
        std::cout << "Epoch loss: " << meanEpochLoss << std::endl;
        
        // Validation 

        // TODO
    }
}

std::tuple<float, float> Network::trainOnBatch(const std::vector<PicData>& batch)
{
    int totalBatchCorrect = 0;
    float totalBatchLoss = 0.f;
    
    #pragma omp parallel for
    for (const auto& pic : batch)
    {   
        auto input = pic.getMat();
        auto label = pic.getLabel();
        std::vector<Matrix> inputs = { input };
        
        // Forward
        for (auto layer : layers)
        {
            inputs.emplace_back(layer->forward(inputs.back()));
        }
        const auto& output = inputs.back();

        // Accuracy
        if (correctPrediction(output, label))
        {
            #pragma omp critical
            {
                totalBatchCorrect += 1;
            }
        }

        // Loss
        auto losses = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
        #pragma omp critical
        {
            for (int i = 0; i < losses.getRows(); i++)
            {
                totalBatchLoss += losses(i, 0);
            }
        }            

        // Gradient
        auto gradient = ErrorFunc::gradSoftmaxCrossentropyWithLogits(output, label);

        // Backward
        for (int i = layers.size() - 1; i >= 0; --i)
        {   
            gradient = layers[i]->backward(inputs[i], gradient);
        }
    }

    // Learn something
    for (auto layer : layers)
    {
        layer->updateWeights(0.001);
    }

    // Return accuracy and loss
    float batchAccuracy = (float)totalBatchCorrect / (float)batch.size();
    float meanBatchLoss = totalBatchLoss / (float)batch.size();
    return std::make_tuple(batchAccuracy, meanBatchLoss);
}

bool Network::correctPrediction(const Matrix& pred, const Matrix& label)
{
    float maxPred = -FLT_MAX;
    int maxIndex = 0;
    int labelIndex = 0;
    for (int i = 0; i < 10; i++)
    {
        if (pred(0, i) > maxPred)
        {
            maxPred = pred(0, i);
            maxIndex = i;
        }
        if (label(0, i) == 1)
        {
            labelIndex = i;
        }

    }
    if (maxIndex == labelIndex)
    {
        return true;
    }
    return false;
}

