#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>
#include <unistd.h>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <omp.h>

#include "MathLib.hpp"
#include "DataLoad.hpp"
#include "NNLib.hpp"

using namespace mathlib;
using namespace mathlib::activation;
using namespace dataload;
using namespace nnlib;

std::vector<std::shared_ptr<ILayer>> buildNetwork()
{
    std::cout << "======================================\n";
    std::cout << "Network architecture\n";

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 1, 28*28);
    std::cout << "Name: " << input->getName() << std::endl;
    std::cout << "Height: " << input->getOutputHeight() << std::endl;
    std::cout << "Width: " << input->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden1", input, 120);
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden1", dense1, std::make_shared<ReLU>());
    std::cout << "Name: " << activation1->getName() << std::endl;
    std::cout << "Height: " << activation1->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_hidden2", activation1, 50);
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_hidden2", dense2, std::make_shared<ReLU>());
    std::cout << "Name: " << activation2->getName() << std::endl;
    std::cout << "Height: " << activation2->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation2->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense3 = std::make_shared<Dense>("Dense_layer_output", activation2, 10);
    std::cout << "Name: " << dense3->getName() << std::endl;
    std::cout << "Height: " << dense3->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense3->getOutputWidth() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1, 
        dense2,
        activation2,
        dense3
    };

    std::cout << "End network architecture\n";
    std::cout << "======================================\n";

    return layers;
}

void getNPics(int n, const std::vector<PicData>& data,
                             std::vector<Matrix>& pics, std::vector<Matrix>& labels)
{
    for (int i = 0; i < n; i++)
    {
        pics.emplace_back(data[i].getMat());
        labels.emplace_back(data[i].getLabel());
    }
}

bool correctPrediction(const Matrix& pred, const Matrix label)
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


int main(int argc, char *argv[])
{  
    
    bool executeTraining = false;
    if (argc == 2 && argv[1] == std::string("-t"))
    {
        executeTraining = true;
    }
    DataLoader dl = DataLoader("../data/fashion_mnist_train_vectors.csv",
                               "../data/fashion_mnist_train_labels.csv");

    // Change the leftmost number here for taking data out

    std::vector<PicData> data = dl.loadNOfEach(6000, 1, 784);
    std::random_shuffle(data.begin(), data.end(),[&](int i) {return std::rand() % i;} );
    //auto validTrainData = dl.getValidTrain(1, 784);
    auto [validationData, trainData] = PreprocessingUtils::splitDataValidTrain(0.1, data);
    //std::vector<PicData> dataPic = std::get<1>(validTrainData);
    // std::vector<Matrix> pics;
    // std::vector<Matrix> labels;
    // getNPics(10, dataPic, pics, labels);

    auto layers = buildNetwork();

    //std::cout << "just printing inputs : "<< dataPic.size() << std::endl;
    // for (int i = 0; i < dataPic.size(); i++) 
    // {
        
    //     // std::cout << pics[i] << std::endl;
    //     std::cout << "this is label: " << std::endl;
    //     std::cout << dataPic[i].getLabel() << std::endl;
        
    // }

    if (!executeTraining)
    {
        return 0;
    }

    // set number of threads
    omp_set_num_threads(8);

    std::cout << "=====================================\n";
    std::cout << "Training\n";

    int epoch = 0;
    int datasetIndex = 0;
    auto totalStartTraining = std::chrono::steady_clock::now();
    while (true)
    {
        auto startTraining = std::chrono::steady_clock::now();
        datasetIndex = 0;
        const int numOfEpochs = 2;
        float epochError = 0.f;
        int batchIndex = 0;
        int epochCorrect = 0;
        if (epoch == numOfEpochs)
        {
            break;
        }

        std::cout << "------------------------------------------\n";
        std::cout << "Epoch no." << ++epoch << std::endl;

        //std::cout << dataPic.size() << "<--- datapic size" << std::endl;
    
        // cycle for batches
        while(datasetIndex < (int)trainData.size())
        {
            //std::cout << datasetIndex << "<--- batch index" << std::endl;
            const int batchSize = 100;
            float batchError = 0.f;
            int batchCorrect = 0;
            //One Batch
            #pragma omp parallel for
            for (int picIndex = 0; picIndex < batchSize; picIndex++)
            {
                // TODO
                // if (datasetIndex + picIndex >= dataPic.size())
                // {
                //     break;
                // }

                Matrix label = trainData[datasetIndex + picIndex].getLabel();
                Matrix input = trainData[datasetIndex + picIndex].getMat();
                
                // Forward
                std::vector<Matrix> inputs = { input };
                for (auto layer : layers)
                {
                    inputs.emplace_back(layer->forward(inputs.back()));
                }

                const auto& output = inputs.back();
                if (correctPrediction(output, label))
                {
                    batchCorrect += 1;
                }
                auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
                #pragma omp critical
                {
                    for (int i = 0; i < errors.getRows(); i++)
                    {
                        batchError += errors(i, 0);
                    }
                }
                //std::cout << "Error " << errors(0, 0) << std::endl;              

                auto grad = ErrorFunc::gradSoftmaxCrossentropyWithLogits(output, label);
                
                // Backward
                for (int i = layers.size() - 1; i >= 0; --i)
                {   
                    grad = layers[i]->backward(inputs[i], grad);
                }
            }
            datasetIndex += batchSize;
            batchIndex += 1;
            auto help = (float)trainData.size()/(float)batchSize;
            for (auto layer : layers)
            {
                layer->updateWeights(epoch , help * (0.002/help) - (float)batchIndex * (0.002/help) );
            }

            std::cout << batchError / (float)batchSize << ", " << std::flush;
            std::cout << batchCorrect / (float)batchSize<< ", "<< std::flush;
            epochError += batchError;
            epochCorrect += batchCorrect;
        }
        auto endTraining = std::chrono::steady_clock::now();
        std::cout << std::endl;
        std::cout << "Elapsed time in training Epoch no. " << epoch << " : "
        << std::chrono::duration_cast<std::chrono::seconds>(endTraining - startTraining).count()
        << " sec" << std::endl;

        float meanEpochError = epochError / (float)trainData.size();
        std::cout << "Epoch error: " << meanEpochError << std::endl;
        float accuracyEpoch = epochCorrect / (float)trainData.size();
        std::cout << "Epoch accuracy: " << accuracyEpoch << std::endl;
        std::cout <<"============================================" << std::endl << "Epoch no. " << epoch << " validation:" << std::endl;
        float error = 0.f;
        float correctPred = 0;
        for (auto dpic : validationData)
        {
            Matrix label = dpic.getLabel();
            Matrix output = dpic.getMat();
            for (auto layer : layers)
            {
                output = layer->forward(output);
            }

            auto probability = ErrorFunc::softMax(output);
            auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
            //auto probability = output;
            for (int i = 0; i < errors.getRows(); i++)
            {
                error += errors(i, 0);
            }
            
            //std::cout << "ErrorFunc::softmaxCrossentropyWithLogits(output, label)" << std::endl;
            //std::cout << ErrorFunc::softmaxCrossentropyWithLogits(output, label);
            //std::cout << "---------------\n";
            //std::cout << "Input: " << dpic.getMat();
            //std::cout << "Prediction: " << probability << std::endl;
            //std::cout << "Label: " << label;
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
    auto totalEndTraining = std::chrono::steady_clock::now();
    std::cout << "Total elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(totalEndTraining - totalStartTraining).count()
    << " sec" << std::endl;
    std::cout << "-------------------------------------\n";
    std::cout << "End training\n";
    std::cout << "=====================================\n";
    
    // std::cout << "=====================================\n";
    // std::cout << "Prediction\n";

    // float error = 0.f;
    // float correctPred = 0;
    // for (auto dpic : dataPic)
    // {
    //     Matrix label = dpic.getLabel();
    //     Matrix output = dpic.getMat();
    //     for (auto layer : layers)
    //     {
    //         output = layer->forward(output);
    //     }

    //     auto probability = ErrorFunc::softMax(output);
    //     auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
    //     //auto probability = output;
    //     for (int i = 0; i < errors.getRows(); i++)
    //     {
    //         error += errors(i, 0);
    //     }
         
    //     //std::cout << "ErrorFunc::softmaxCrossentropyWithLogits(output, label)" << std::endl;
    //     std::cout << ErrorFunc::softmaxCrossentropyWithLogits(output, label);
    //     std::cout << "---------------\n";
    //     //std::cout << "Input: " << dpic.getMat();
    //     std::cout << "Prediction: " << probability << std::endl;
    //     std::cout << "Label: " << label;
    //     if (correctPrediction(probability, label))
    //     {
    //         correctPred += 1.f;
    //     }
    // }

    // std::cout << "Prediction error " << error << std::endl;
    // std::cout << "percentage correct: " << std::endl;
    // std::cout << (correctPred/((float)dataPic.size()) )*100.f << "%" << std::endl;
    // std::cout << "End prediction\n";
    // std::cout << "=====================================\n";
    return 0 ;
}
