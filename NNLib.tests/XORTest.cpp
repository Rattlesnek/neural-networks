#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <string>

#include "MathLib.hpp"
#include "NNLib.hpp"

using namespace mathlib;
using namespace mathlib::activation;
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

    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden", dense1, std::make_shared<Sigmoid>());
    std::cout << "Name: " << activation1->getName() << std::endl;
    std::cout << "Height: " << activation1->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_output", activation1, 1);
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_output", dense2, std::make_shared<Sigmoid>());
    std::cout << "Name: " << activation2->getName() << std::endl;
    std::cout << "Height: " << activation2->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation2->getOutputWidth() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1, 
        dense2,
        activation2
    };

    std::cout << "End network architecture\n";
    std::cout << "======================================\n";

    return layers;
}


TEST(XORTest, Integration)
{
    // Dataset
    // tuple(XOR-input, one-hot-vector)
    std::vector<std::tuple<Matrix, Matrix>> dataXOR = {
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {0, 0}), Matrix(1, 1, {0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {0, 1}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {1, 0}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 2, {1, 1}), Matrix(1, 1, {0}))
    };

    // Create vector of layers
    auto layers = buildNetwork();

    std::cout << "=====================================\n";
    std::cout << "Training\n";

    int iterCnt = 0;
    while (true)
    {   
        if (iterCnt == 5000)
        {
            break;
        }

        std::cout << "Iteration no." << ++iterCnt << "\t";
        float totalError = 0.f;
        for (auto [input, label] : dataXOR)
        {            
            // Forward
            auto output = input;
            for (auto layer : layers)
            {
                output = layer->forward(output);
            }
            
            totalError += ErrorFunc::meanSquareError(output, label);

            // Backward       
            auto grad = output - label;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {   
                auto layer = *it;
                grad = layer->backward(grad);
            }
        }
        // Weight update
        for (auto layer : layers)
        {
            layer->updateWeights();
        } 

        std::cout << "Total Error: " << totalError << std::endl;
    }

    std::cout << "End training\n";
    std::cout << "=====================================\n";

    std::cout << "=====================================\n";
    std::cout << "Prediction\n";

    float predictionError = 0.f;
    for (auto [input, label] : dataXOR)
    {
        auto output = input;
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }
        predictionError += ErrorFunc::meanSquareError(output, label);

        std::cout << "---------------\n";
        std::cout << "Input: " << input;
        std::cout << "Prediction: " << output;
        std::cout << "Label: " << label;
    }
    
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";
    std::cout << "Prediction Error: " << predictionError << std::endl;
    std::cout << "=====================================\n";
    EXPECT_TRUE(predictionError < 2);
}
