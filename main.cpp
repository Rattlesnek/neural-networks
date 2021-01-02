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

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 1, 28*28);
    
    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_hidden1", input, 120);
    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_hidden1", dense1, std::make_shared<LeakyReLU>());

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_hidden2", activation1, 90);
    std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_hidden2", dense2, std::make_shared<LeakyReLU>());

     std::shared_ptr<ILayer> dense3 = std::make_shared<Dense>("Dense_layer_hidden2", activation2, 60);
     std::shared_ptr<ILayer> activation3 = std::make_shared<Activation>("Activation_hidden2", dense3, std::make_shared<LeakyReLU>());

    std::shared_ptr<ILayer> dense4 = std::make_shared<Dense>("Dense_layer_hidden2", activation3, 30);
    std::shared_ptr<ILayer> activation4 = std::make_shared<Activation>("Activation_hidden2", dense4, std::make_shared<LeakyReLU>());

    std::shared_ptr<ILayer> output = std::make_shared<Dense>("Dense_layer_output", activation4, 10);

    std::vector<std::shared_ptr<ILayer>> layers = {
        input,
        dense1,
        activation1, 
        dense2,
        activation2,
        dense3,
        activation3,
        dense4,
        activation4,
        output
    };

    for (auto layer : layers)
    {
        std::cout << "Name: " << layer->getName() << std::endl;
        std::cout << "Height: " << layer->getOutputHeight() << std::endl;
        std::cout << "Width: " << layer->getOutputWidth() << std::endl;
    }

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

    auto layers = buildNetwork();
    Network network(layers);

    if (!executeTraining)
    {
        return 0;
    }   

    // set number of threads
    omp_set_num_threads(8);

    DataLoader trainDataLoader = DataLoader("../data/fashion_mnist_train_vectors.csv",
                                            "../data/fashion_mnist_train_labels.csv");

    DataLoader testDataLoader = DataLoader("../data/fashion_mnist_test_vectors.csv",
                                           "../data/fashion_mnist_test_labels.csv");

    // Change the leftmost number here for taking data out
    auto allTrainData = trainDataLoader.loadNOfEach(600, 1, 784);
    
    // auto allTrainData = trainDataLoader.loadAllData(1, 784);
    // const auto& trainData = allTrainData;
    // auto testData = testDataLoader.loadAllData(1, 784);

    std::random_shuffle(allTrainData.begin(), allTrainData.end(), [&](int i){ return std::rand() % i; } );
    
    auto [testData, trainData] = PreprocessingUtils::splitDataValidTrain(0.1, allTrainData);

    float learningRate = 0.0005;
    float momentumFactor = 0.9;

    network.train(2, 100, learningRate, momentumFactor, trainData, testData);

    std::vector<Matrix> mats;
    for (const auto& pic : testData)
    {
        mats.emplace_back(pic.getMat());
    }

    auto predictionOutputs = network.predict(mats);

    // for (const auto& output : predictionOutputs)
    // {
    //     std::cout << output;
    // }
    

    return 0;
}
