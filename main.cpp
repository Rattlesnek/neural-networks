#include <iostream>
#include "MathLib.hpp"
#include "DataLoad.hpp"
#include "NNLib.hpp"

#include <memory>
#include <algorithm>

using namespace mathlib;
using namespace mathlib::activation;
using namespace dataload;
using namespace nnlib;

int main()
{  
    // tuple(XOR-input, one-hot-vector)
    std::vector<std::tuple<Matrix, Matrix>> dataXOR = {
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {0, 0}), Matrix(1, 1, {0})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {0, 1}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {1, 0}), Matrix(1, 1, {1})),
        std::tuple<Matrix, Matrix>(Matrix(2, 1, {1, 1}), Matrix(1, 1, {0}))
    };

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 2, 1);
    std::cout << "Name: " << input->getName() << std::endl;
    std::cout << "Type: " << input->getType() << std::endl;
    std::cout << "Height: " << input->getOutputHeight() << std::endl;
    std::cout << "Width: " << input->getOutputWidth() << std::endl;
    std::cout << "NeuronOutput: " << input->getNeuronOutput() << std::endl;

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden", LayerType::HiddenLayer, 2, input, std::make_shared<ReLU>());
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Type: " << dense1->getType() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;
    std::cout << "NeuronOutput: " << dense1->getNeuronOutput() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_output", LayerType::OutputLayer, 1, dense1, std::make_shared<Sigmoid>());
    std::cout << "Name: " << dense2->getName() << std::endl;
    std::cout << "Type: " << dense2->getType() << std::endl;
    std::cout << "Height: " << dense2->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense2->getOutputWidth() << std::endl;
    std::cout << "NeuronOutput: " << dense2->getNeuronOutput() << std::endl;

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1, 
        dense2
    };

    std::cout << "=====================================\n";
    std::cout << "dataset single pass\n";
    for (const auto& tuple : dataXOR)
    {
        auto input = std::get<0>(tuple);
        auto label = std::get<1>(tuple);
        std::cout << "input: " << input << std::endl;
        std::cout << "label: " << label << std::endl;

        std::cout << "------------------------------------------\n";
        std::cout << "forward pass\n";
        auto output = input;
        for (auto layer : layers)
        {
            std::cout << layer->getName() << std::endl;
            
            output = layer->forward(output);
            std::cout << output << std::endl;
        }

        std::cout << "end forward pass\n";
        std::cout << "------------------------------------------\n";
        

        // std::cout << "------------------------------------------\n";
        // std::cout << "backward pass\n";
        // label.applyFunc([](float x) -> float { return -x; });
        // auto error = output + label;
        // std::cout << error << std::endl;
        
        // auto it = layers.rbegin();
        // std::shared_ptr<ILayer> layer = *it;
        // for (; it != layers.rend(); ++it, layer = *it)
        // {
        //     std::cout << layer->getName() << std::endl;
            
        //     error = layer->backward(error);
        //     std::cout << error << std::endl;
        // }

        // std::cout << "end backward pass\n";
        // std::cout << "------------------------------------------\n";
        

        // TODO tmp break
        //break;
    }

    return 0;
}