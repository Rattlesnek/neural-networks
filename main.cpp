#include <iostream>
#include <memory>
#include <algorithm>
#include <string>

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

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 2, 1);
    std::cout << "Name: " << input->getName() << std::endl;
    std::cout << "Type: " << input->getType() << std::endl;
    std::cout << "Height: " << input->getOutputHeight() << std::endl;
    std::cout << "Width: " << input->getOutputWidth() << std::endl;
    //std::cout << "NeuronOutput: " << input->getLastOutput() << std::endl;

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden", LayerType::HiddenLayer, 2, input, std::make_shared<Sigmoid>());
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Type: " << dense1->getType() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;
    //std::cout << "NeuronOutput: " << dense1->getLastOutput() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_output", LayerType::OutputLayer, 1, dense1, std::make_shared<Sigmoid>());
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Type: " << dense2->getType() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;
    //std::cout << "NeuronOutput: " << dense2->getLastOutput() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1, 
        dense2
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
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {0, 0}), Matrix(1, 1, {0})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {0, 1}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {1, 0}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {1, 1}), Matrix(1, 1, {0}))
    };

    // Create vector of layers
    auto layers = buildNetwork();

    if (!executeTraining)
    {
        return 0;
    }

    std::cout << "=====================================\n";
    std::cout << "Training\n";

    int iterCnt = 0;

    while (true)
    {   
        if (iterCnt == 5000)
        {
            break;
        }

        std::cout << "------------------------------------------\n";
        std::cout << "Iteration no." << ++iterCnt << " ";

        float totalError = 0.f;
        for (auto [input, label] : dataXOR)
        {
            // std::cout << "input: " << input << std::endl;
            // std::cout << "label: " << label << std::endl;

            // std::cout << "------------------------------------------\n";
            // std::cout << "Forward pass\n";
            
            auto output = input;
            for (auto layer : layers)
            {
                //std::cout << "  " << layer->getName() << std::endl;
                output = layer->forward(output);
                //std::cout << output << std::endl;
            }

            // std::cout << "End forward pass\n";
            // std::cout << "------------------------------------------\n";
            
            totalError += ErrorFunc::meanSquareError(output, label);
            
            // std::cout << "------------------------------------------\n";
            // std::cout << "Backward pass\n"; 
            
            auto grad = output - label;
            //std::cout << grad << std::endl;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {   
                auto layer = *it;
                //std::cout << "  " << layer->getName() << std::endl;
                grad = layer->backward(grad);
                //std::cout << grad << std::endl;
            }

            // std::cout << "End backward pass\n";
            // std::cout << "------------------------------------------\n";
        }

        std::cout << "Total Error: " << totalError << std::endl;
    }

    std::cout << "------------------------------------------\n";
    std::cout << "End training\n";
    std::cout << "=====================================\n";

    std::cout << "=====================================\n";
    std::cout << "Prediction\n";

    for (auto [input, label] : dataXOR)
    {
        auto output = input;
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }
        std::cout << "---------------\n";
        std::cout << "Input: " << input;
        std::cout << "Prediction: " << output;
        std::cout << "Label: " << label;
    }
    
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";

    return 0;
}
