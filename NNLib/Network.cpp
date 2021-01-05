#include "Network.h"
#include <memory>
#include <chrono>
#include <cfloat>

#include "MathLib.hpp"
#include "NetworkException.h"
#include "TrainUtils.h"

using namespace nnlib;
using namespace mathlib;
using namespace dataload;

Network::Network(std::vector<std::shared_ptr<ILayer>> layers) :
    layers(std::move(layers))
{

}

// std::optional<std::vector<PicData>&> was giving troubles so i just didnt do it that way yet
void Network::train(const int numOfEpochs, 
    const int batchSize, 
    const float learningRate, 
    const float momentumFactor, 
    const std::vector<PicData>& trainData, 
    std::optional<std::vector<dataload::PicData>> validationData)
{
    if (trainData.size() % batchSize != 0)
    {
        throw NetworkException("TrainData size is not divisible by batchSize.");
    }
    int numOfBatches = trainData.size() / batchSize;
    int maxBatches = numOfBatches * numOfEpochs;
    int currentBatch = 0;
    
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

        std::cout << "Mean Batch Loss: " << std::flush;
        // Training
        int datasetIndex = 0;
        while(datasetIndex < (int)trainData.size())
        {           
            std::vector<PicData> batch(trainData.begin() + datasetIndex, trainData.begin() + datasetIndex + batchSize);

            float alpha = TrainUtils::piecewiseScheduling(learningRate, maxBatches, currentBatch);

            auto [batchAccuracy, meanBatchLoss] = trainOnBatch(batch, alpha, momentumFactor);

            std::cout << meanBatchLoss << ", " << std::flush;

            totalBatchAccuracy += batchAccuracy;
            totalMeanBatchLoss += meanBatchLoss;
            datasetIndex += batchSize;
            currentBatch += 1;
        }

        auto epochEndTime = std::chrono::steady_clock::now();
        std::cout << std::endl << "Elapsed time in training Epoch no. " << epoch << " : "
            << std::chrono::duration_cast<std::chrono::seconds>(epochEndTime - epochStartTime).count() << " sec" << std::endl;

        float epochAccuracy = totalBatchAccuracy / (float)numOfBatches;
        std::cout << "Epoch accuracy: " << epochAccuracy * 100 << std::endl;

        float meanEpochLoss = totalMeanBatchLoss / (float)numOfBatches;
        std::cout << "Epoch loss: " << meanEpochLoss << std::endl;
        
        // Validation 
        if (validationData.has_value())
        {
            auto [validAccuracy, meanValidLoss] = validate(*validationData);
            std::cout << "------------------------------------------\n";
            std::cout << "Validation\n";
            std::cout << "Validation accuracy: " << validAccuracy * 100 << std::endl;
            std::cout << "Validation loss: " << meanValidLoss << std::endl;
        }
        std::cout << "------------------------------------------\n";
    }

    auto totalEndTime = std::chrono::steady_clock::now();

    std::cout << "Total elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(totalEndTime - totalStartTime).count()
    << " sec" << std::endl;
    std::cout << "=====================================\n";
}

std::tuple<float, float> Network::trainOnBatch(const std::vector<PicData>& batch, const float alpha, const float momentumFactor)
{
    int totalBatchCorrect = 0;
    float totalBatchLoss = 0.f;
    #pragma omp parallel for
    for (const auto& pic : batch)
    {   
        auto input = pic.getMat();
        auto label = pic.getLabels();
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

    for (auto layer : layers)
    {
        layer->updateWeights(alpha, momentumFactor);
    }

    // Return accuracy and loss
    float batchAccuracy = (float)totalBatchCorrect / (float)batch.size();
    float meanBatchLoss = totalBatchLoss / (float)batch.size();
    return std::make_tuple(batchAccuracy, meanBatchLoss);
}

std::tuple<float, float> Network::validate(const std::vector<dataload::PicData>& validationData)
{
    float totalValidLoss = 0.f;
    int correctPred = 0;

    #pragma omp parallel for
    for (const auto& pic : validationData)
    {
        auto label = pic.getLabels();
        auto output = pic.getMat();
        // Forward
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }

        auto probability = ErrorFunc::softMax(output);

        auto losses = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
        
        #pragma omp critical
        {
            for (int i = 0; i < losses.getRows(); i++)
            {
                totalValidLoss += losses(i, 0);
            }
        }
        
        if (correctPrediction(probability, label))
        {
            #pragma omp critical
            {
                correctPred += 1;
            }
        }
    }

    float validAccuracy = (float) correctPred / (float) validationData.size();
    float meanValidLoss = (float) totalValidLoss / (float) validationData.size();
    return std::make_tuple(validAccuracy, meanValidLoss);
}

std::vector<Matrix> Network::predict(const std::vector<Matrix>& predictionInputs)
{
    std::vector<Matrix> outputProbabilities;
    
    for (const auto& input : predictionInputs)
    {
        // Forward
        Matrix output = input;
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }
        auto probability = ErrorFunc::softMax(output);
        
        outputProbabilities.emplace_back(probability);
    }

    return outputProbabilities;
}

int Network::findMaxIndex(const Matrix& pred)
{
    float maxPred = -FLT_MAX;
    int maxIndex = -1;
    for (int i = 0; i < 10; i++)
    {
        if (pred(0, i) > maxPred)
        {
            maxPred = pred(0, i);
            maxIndex = i;
        }
    }
    return maxIndex;
}

bool Network::correctPrediction(const Matrix& pred, const std::vector<int>& labels)
{
    int maxIndex = findMaxIndex(pred);
    return (maxIndex == labels[0]);
}
