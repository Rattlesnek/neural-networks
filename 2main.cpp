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

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 1, 28*28);
    std::cout << "Name: " << input->getName() << std::endl;
    std::cout << "Height: " << input->getOutputHeight() << std::endl;
    std::cout << "Width: " << input->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden1", input, 4);
    std::cout << "Name: " << dense1->getName() << std::endl;
    std::cout << "Height: " << dense1->getOutputHeight() << std::endl;
    std::cout << "Width: " << dense1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden1", dense1, std::make_shared<ReLU>());
    std::cout << "Name: " << activation1->getName() << std::endl;
    std::cout << "Height: " << activation1->getOutputHeight() << std::endl;
    std::cout << "Width: " << activation1->getOutputWidth() << std::endl;

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_hidden2", activation1, 5);
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


int main(int argc, char *argv[])
{  
    
    bool executeTraining = false;
    if (argc == 2 && argv[1] == std::string("-t"))
    {
        executeTraining = true;
    }
    DataLoader dl = DataLoader("/home/xkolla/neural-networks/data/fashion_mnist_train_vectors.csv",
                                             "/home/xkolla/neural-networks/data/fashion_mnist_train_labels.csv");
    std::vector<PicData> dataPic = dl.loadNData(2000, 1, 784);
    std::vector<Matrix> pics;
    std::vector<Matrix> labels;
    getNPics(10, dataPic, pics, labels);
    auto layers = buildNetwork();
    // std::cout << "just testing outputs:" << std::endl;
    // for (int i = 0; i < 2; i++) 
    // {
    //     std::cout << (dataPic[i].getMat() == pics[i]) << std::endl;
    //     std::cout << "this is dataPics mat: " << std::endl;
    //     std::cout << dataPic[i].getMat() << std::endl;
    //     std::cout << "this is pics mat: " << std::endl;
    //     std::cout << pics[i] << std::endl;
    //     std::cout << "this is dataPics label: " << std::endl;
    //     std::cout << dataPic[i].getLabel() << std::endl;
    //     std::cout << "this is labels label: " << std::endl;
    //     std::cout << labels[i] << std::endl;
    // }
    if (!executeTraining)
    {
        return 0;
    }
    std::cout << "=====================================\n";
    std::cout << "Training\n";

    int iterations = 0;
    int count = 0;
    while (true)
    {
        if (iterations == 50)
        {
            break;
        }
        std::cout << "------------------------------------------\n";
        std::cout << "Iteration no." << ++iterations << " ";
        float error = 0.f;

        for ( PicData dpic : dataPic)
        {
            if (count == 40)
            {
                break;
            }
            count = count + 1;
            // Forward
            Matrix label = dpic.getLabel();
            Matrix output = dpic.getMat();
            for (auto layer : layers)
            {
                output = layer->forward(output);
            }

            auto errors = ErrorFunc::softmaxCrossentropyWithLogits(output, label);
            //auto probability = output;
            for (int i = 0; i < errors.getRows(); i++)
            {
                error += errors(i, 0);
            }
            std::cout << "error variable; softmaxCrosseentropy with logits added:" << std::endl;
            std::cout << error << std::endl;
            auto grad = ErrorFunc::gradSoftmaxCrossentropyWithLogits(output, label);
            
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {   
                auto layer = *it;
                grad = layer->backward(grad);
            }
        }
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
    int count1 = 0;
    for (auto dpic : dataPic)
    {
        if (count1 == 40)
        {
            break;
        }
        count1 = count1+1;
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
         
        std::cout << "ErrorFunc::softmaxCrossentropyWithLogits(output, label)" << std::endl;
        std::cout << ErrorFunc::softmaxCrossentropyWithLogits(output, label);
        std::cout << "---------------\n";
        //std::cout << "Input: " << dpic.getMat();
        std::cout << "Prediction: " << probability << std::endl;
        std::cout << "Label: " << label;
    }

    std::cout << "Prediction error " << error << std::endl;
    
    std::cout << "End prediction\n";
    std::cout << "=====================================\n";
    return 0 ;
}
