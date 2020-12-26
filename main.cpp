#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <omp.h>
#include <unistd.h>

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

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 1, 2);
    std::cout << "Name: " << input->getName() << std::endl;
    std::cout << "Height: " << input->getOutputHeight() << std::endl;
    std::cout << "Width: " << input->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden", input, 2);
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden", dense1, std::make_shared<ReLU>());
    std::cout << "Name: " << activation1->getName() << std::endl;
    std::cout << "Height: " << activation1->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_output", activation1, 2);
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;

    // std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_output", dense2, std::make_shared<Sigmoid>());
    // std::cout << "Name: " << activation2->getName() << std::endl;
    // std::cout << "Height: " << activation2->getOutputHeight() << std::endl;
    // std::cout << "Width: " << activation2->getOutputWidth() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1,
        dense2,
        //activation2
    };

    std::cout << "End network architecture\n";
    std::cout << "======================================\n";

    return layers;
}


int main(int argc, char *argv[])
{  
    bool executeTraining = false;
    if (argc == 2 && argv[1] == std::string("-t"))
    {
        executeTraining = true;
    }

    // Dataset
    // tuple(XOR-input, one-hot-vector)
    std::vector<std::tuple<Matrix, Matrix>> dataXOR = {
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {0, 0}), Matrix(1, 2, {0, 1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {0, 1}), Matrix(1, 2, {1, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {1, 0}), Matrix(1, 2, {1, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {1, 1}), Matrix(1, 2, {0, 1}))
    };

    // Create vector of layers
    auto layers = buildNetwork();

    if (!executeTraining)
    {
        return 0;
    }    

    std::cout << "=====================================\n";
    std::cout << "Training\n";

    // set number of threads
    omp_set_num_threads(4);

    int iterCnt = 0;

    while (true)
    {   
        if (iterCnt == 10000)
        {
            break;
        }

        std::cout << "------------------------------------------\n";
        std::cout << "Iteration no." << ++iterCnt << " ";
        
        float error = 0.f;
        #pragma omp parallel for
        for (int i = 0; i < (int)dataXOR.size(); i++)
        {
            // Forward
            auto label = std::get<1>(dataXOR[i]);
            std::vector<Matrix> inputs = { std::get<0>(dataXOR[i]) };
            for (auto layer : layers)
            {
                inputs.emplace_back(layer->forward(inputs.back()));
            }

            const auto& output = inputs.back(); 
            #pragma omp critical
            {
                error += ErrorFunc::softmaxCrossentropyWithLogits(output, label)(0,0);
            }
            auto grad = ErrorFunc::gradSoftmaxCrossentropyWithLogits(output, label);
            
            // Backward
            for (int i = layers.size() - 1; i >= 0; --i)
            {   
                grad = layers[i]->backward(inputs[i], grad);
            }        
        }
        // Weight update
        for (auto layer : layers)
        {
            layer->updateWeights();
        } 

        std::cout << "Error: " << error << std::endl;
    }

    std::cout << "------------------------------------------\n";
    std::cout << "End training\n";
    std::cout << "=====================================\n";

    std::cout << "=====================================\n";
    std::cout << "Prediction\n";

    float error = 0.f;
    for (auto [input, label] : dataXOR)
    {
        auto output = input;
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }

        auto probability = ErrorFunc::softMax(output);
        error += ErrorFunc::softmaxCrossentropyWithLogits(output, label)(0,0);
        std::cout << "--------------------------\n";
        std::cout << "Input: " << input;
        std::cout << "Prediction: " << probability;
        std::cout << "Label: " << label;
    }

    std::cout << "Prediction error " << error << std::endl;
    
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";

    return 0;
}
