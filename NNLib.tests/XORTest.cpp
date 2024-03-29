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

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_output", activation1, 2);
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1,
        dense2,
    };

    std::cout << "End network architecture\n";
    std::cout << "======================================\n";

    return layers;
}

TEST(XORTest, XOR_2D_Integration)
{  

    // Dataset
    // tuple(XOR-input, one-hot-vector)
    std::vector<std::tuple<Matrix, std::vector<int>>> dataXOR = {
        std::tuple<Matrix, std::vector<int>>(Matrix(1, 2, {0, 0}), {1}),
        std::tuple<Matrix, std::vector<int>>(Matrix(1, 2, {0, 1}), {0}),
        std::tuple<Matrix, std::vector<int>>(Matrix(1, 2, {1, 0}), {0}),
        std::tuple<Matrix, std::vector<int>>(Matrix(1, 2, {1, 1}), {1})
    };

    // Create vector of layers
    auto layers = buildNetwork();  

    std::cout << "=====================================\n";
    std::cout << "Training\n";

    // set number of threads
    // omp_set_num_threads(4);

    int iterCnt = 0;

    while (true)
    {   
        if (iterCnt == 1000)
        {
            break;
        }

        std::cout << "------------------------------------------\n";
        std::cout << "Iteration no." << ++iterCnt << " ";
        
        float error = 0.f;
        #pragma omp parallel for
        for (auto [input, label] : dataXOR)
        {
            // Forward
            std::vector<Matrix> inputs = { input };
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
            layer->updateWeights(0.01, 0);
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
        std::cout << "Label: " << label[0] << std::endl;
    }

    std::cout << "Prediction error " << error << std::endl;
    
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";

    EXPECT_TRUE(error < 0.001f);
}

TEST(XORTest, XOR_3D_Integration)
{  
    // Dataset
    // tuple(XOR-input, one-hot-vector)
    std::vector<std::tuple<Matrix, Matrix>> dataXOR = {
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {0, 0, 0}), Matrix(1, 3, {1, 0, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {0, 0, 1}), Matrix(1, 3, {0, 1, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {0, 1, 0}), Matrix(1, 3, {0, 0, 1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {0, 1, 1}), Matrix(1, 3, {0, 0, 1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {1, 0, 0}), Matrix(1, 3, {0, 1, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {1, 0, 1}), Matrix(1, 3, {0, 0, 1})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {1, 1, 0}), Matrix(1, 3, {0, 1, 0})),
        std::tuple<Matrix, Matrix>(Matrix(1, 3, {1, 1, 1}), Matrix(1, 3, {1, 0, 0}))
    };
}
