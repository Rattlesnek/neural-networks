#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>
#include <unistd.h>
#include <numeric>
#include <cmath>
#include <cfloat>

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

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden1", input, 20);
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden1", dense1, std::make_shared<Sigmoid>());
    std::cout << "Name: " << activation1->getName() << std::endl;
    std::cout << "Height: " << activation1->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_hidden2", activation1, 10);
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_hidden2", dense2, std::make_shared<Sigmoid>());
    std::cout << "Name: " << activation2->getName() << std::endl;
    std::cout << "Height: " << activation2->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation2->getOutputWidth() << std::endl;

    // std::shared_ptr<ILayer> dense3 = std::make_shared<Dense>("Dense_layer_hidden3", activation2, 5);
    // std::cout << "Name: " << dense3->getName() << std::endl;
    // std::cout << "Height: " << dense3->getOutputHeight() << std::endl;
    // std::cout << "Width: " << dense3->getOutputWidth() << std::endl;

    // std::shared_ptr<ILayer> activation3 = std::make_shared<Activation>("Activation_hidden3", dense3, std::make_shared<ReLU>());
    // std::cout << "Name: " << activation3->getName() << std::endl;
    // std::cout << "Height: " << activation3->getOutputHeight() << std::endl;
    // std::cout << "Width: " << activation3->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense4 = std::make_shared<Dense>("Dense_layer_output", activation2, 10);
    std::cout << "Name: " << dense4->getName() << std::endl;
    std::cout << "Height: " << dense4->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense4->getOutputWidth() << std::endl;

    // std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_output", dense2, std::make_shared<Sigmoid>());
    // std::cout << "Name: " << activation2->getName() << std::endl;
    // std::cout << "Height: " << activation2->getOutputHeight() << std::endl;
    // std::cout << "Width: " << activation2->getOutputWidth() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1, 
        dense2,
        activation2,
        // dense3,
        // activation3,
        dense4
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
    std::vector<PicData> dataPic = dl.loadNOfEach(20, 1, 784);
    std::random_shuffle(dataPic.begin(), dataPic.end(),[&](int i) {return std::rand() % i;} );
    std::vector<Matrix> pics;
    std::vector<Matrix> labels;
    getNPics(10, dataPic, pics, labels);
    auto layers = buildNetwork();
    std::cout << "just printing inputs : "<< dataPic.size() << std::endl;
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
    std::cout << "=====================================\n";
    std::cout << "Training\n";

    int Epocha = 0;
    int batchIndex = 0;
    
    while (true)
    {
        auto startTraining = std::chrono::steady_clock::now();
        batchIndex = 0;
        if (Epocha == 100)
        {
            break;
        }
        std::cout << "------------------------------------------\n";
        std::cout << "Epocha no." << ++Epocha << std::endl;
        //std::cout << dataPic.size() << "<--- datapic size" << std::endl;
        float error = 0.f;
        // cycle for batches
        while( batchIndex < dataPic.size())
        {
            //std::cout << batchIndex << "<--- batch index" << std::endl;
            for ( int picIndex = 0; picIndex < 20; picIndex ++)
            {
                if(batchIndex + picIndex >= dataPic.size())
                {
                    break;
                }
                // Forward
                Matrix label = dataPic[batchIndex + picIndex].getLabel();
                Matrix output = dataPic[batchIndex + picIndex].getMat();
                for (auto layer : layers)
                {
                    output = layer->forward(output);
                    
                }
                // std::cout << "output after forwards:" << std::endl << output;
                // std::cout << "labels output :" << std::endl << label;
                auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
                //auto probability = output;
                for (int i = 0; i < errors.getRows(); i++)
                {
                    error += errors(i, 0);
                }
                
                // std::cout << "error variable, softmaxCrosseentropy with logits added:" << std::endl;
                // std::cout << error << std::endl;
                // std::cout << "errors :" << std::endl;
                // std::cout << errors ;
                auto grad = ErrorFunc::gradSoftmaxCrossentropyWithLogits(output, label);
                
                // std::cout << "gradient: " << std::endl;
                // std::cout << grad;
                // Backward
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                {   
                    auto layer = *it;
                    grad = layer->backward(grad);
                }
            }
            batchIndex += 20;
            for (auto layer : layers)
            {
                layer->updateWeights(Epocha);
            }
        }
        auto endTraining = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in training Epocha no. " << Epocha << " : "
        << std::chrono::duration_cast<std::chrono::seconds>(endTraining - startTraining).count()
        << " sec" << std::endl;

        // std::cout << "Error: " << error << std::endl;
    }
    
    std::cout << "------------------------------------------\n";
    std::cout << "End training\n";
    std::cout << "=====================================\n";
    
    std::cout << "=====================================\n";
    std::cout << "Prediction\n";

    float error = 0.f;
    float correctPred = 0;
    for (auto dpic : dataPic)
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
        std::cout << ErrorFunc::softmaxCrossentropyWithLogits(output, label);
        std::cout << "---------------\n";
        //std::cout << "Input: " << dpic.getMat();
        std::cout << "Prediction: " << probability << std::endl;
        std::cout << "Label: " << label;
        if (correctPrediction(probability, label))
        {
            correctPred += 1.f;
        }
    }

    std::cout << "Prediction error " << error << std::endl;
    std::cout << "percentage correct: " << std::endl;
    std::cout << (correctPred/((float)dataPic.size()) )*100.f << "%" << std::endl;
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";
    return 0 ;
}
