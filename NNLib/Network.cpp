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

Network::Network(std::vector<std::shared_ptr<ILayer>> layers, float alpha) :
    layers(std::move(layers)), alpha(alpha), totalSteps(0), epoch(0), epochLimit(0)
{

}
// std::optional<std::vector<PicData>&> was giving troubles so i just didnt do it that way yet
void Network::train(const int numOfEpochs, const int batchSize, const std::vector<PicData>& trainData, std::vector<dataload::PicData>& validationData)
{
    epochLimit = numOfEpochs;
    if (trainData.size() % batchSize != 0)
    {
        throw NetworkException("TrainData size is not divisible by batchSize.");
    }
    int numOfBatches = trainData.size() / batchSize;
    
    std::cout << "=====================================\n";
    std::cout << "Training\n";
    
    auto totalStartTime = std::chrono::steady_clock::now();
    
    
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

            auto [batchAccuracy, meanBatchLoss] = trainOnBatch(batch, trainData.size());

            std::cout << batchAccuracy << ", " << std::flush;

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

        Network::predict(validationData);
    }
    auto totalEndTime = std::chrono::steady_clock::now();
    std::cout << "Total elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(totalEndTime - totalStartTime).count()
    << " sec" << std::endl;
    std::cout << "=====================================\n";
}

std::tuple<float, float> Network::trainOnBatch(const std::vector<PicData>& batch, int stepsPerEpocha)
{
    int totalBatchCorrect = 0;
    float totalBatchLoss = 0.f;
    #pragma omp parallel for
    for (const auto& pic : batch)
    {   
        totalSteps++;
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

    // Learn something, viz. TrainUtils , alpha == LR
    // This needs work, (TrainUtils)
    alpha = TrainUtils::powerSchedulingLR(alpha, stepsPerEpocha, totalSteps);
    //alpha = TrainUtils::exponentialScheduling(alpha , stepsPerEpocha , totalSteps);
    //alpha =  TrainUtils::oneCycleScheduling(alpha, stepsPerEpocha * epochLimit , totalSteps);
    for (auto layer : layers)
    {
        layer->updateWeights(alpha);
    }

    // Return accuracy and loss
    float batchAccuracy = (float)totalBatchCorrect / (float)batch.size();
    float meanBatchLoss = totalBatchLoss / (float)batch.size();
    return std::make_tuple(batchAccuracy, meanBatchLoss);
}

void Network::predict(std::vector<dataload::PicData> validationData)
{
    float error = 0.f;
    float correctPred = 0;
    for (auto dpic : validationData)
        {
            std::vector<int> label = dpic.getLabels();
            Matrix output = dpic.getMat();
            for (auto layer : layers)
            {
                output = layer->forward(output);
            }

            auto probability = ErrorFunc::softMax(output);

            auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
            
            for (int i = 0; i < errors.getRows(); i++)
            {
                error += errors(i, 0);
            }
            if (correctPrediction(probability, label))
            {
                correctPred += 1.f;
            }
            
        }
        std::cout << "Mean prediction error: " << error/(float) validationData.size() << std::endl;
        std::cout << "Percentage correct: " ;
        std::cout << (correctPred/((float)validationData.size()) )*100.f << "%" << std::endl;
        std::cout << "End validation\n";
        std::cout << "=====================================\n";
}

bool Network::correctPrediction(const Matrix& pred, const std::vector<int>& labels)
{
    float maxPred = -FLT_MAX;
    int maxIndex = 0;
    for (int i = 0; i < 10; i++)
    {
        if (pred(0, i) > maxPred)
        {
            maxPred = pred(0, i);
            maxIndex = i;
        }
    }
    if (maxIndex == labels[0])
    {
        return true;
    }
    return false;
}


