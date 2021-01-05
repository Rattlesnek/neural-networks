#include <iostream>
#include <memory>
#include <string>
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
    std::cout << "=====================================\n";
    std::cout << "Network architecture\n";

    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_layer", 1, 28*28);
    
    std::shared_ptr<ILayer> dense1 = std::make_shared<Dense>("Dense_layer_1", input, 120);
    std::shared_ptr<ILayer> activation1 = std::make_shared<Activation>("Activation_1", dense1, std::make_shared<LeakyReLU>());

    std::shared_ptr<ILayer> dense2 = std::make_shared<Dense>("Dense_layer_2", activation1, 90);
    std::shared_ptr<ILayer> activation2 = std::make_shared<Activation>("Activation_2", dense2, std::make_shared<LeakyReLU>());

     std::shared_ptr<ILayer> dense3 = std::make_shared<Dense>("Dense_layer_3", activation2, 60);
     std::shared_ptr<ILayer> activation3 = std::make_shared<Activation>("Activation_3", dense3, std::make_shared<LeakyReLU>());

    std::shared_ptr<ILayer> dense4 = std::make_shared<Dense>("Dense_layer_4", activation3, 30);
    std::shared_ptr<ILayer> activation4 = std::make_shared<Activation>("Activation_4", dense4, std::make_shared<LeakyReLU>());

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
    std::cout << "=====================================\n";
    return layers;
}

int main(int argc, char *argv[])
{  
    // Set number of threads
    omp_set_num_threads(8);
    
    // Create network
    auto layers = buildNetwork();
    Network network(layers);

    // Load train data
    DataLoader trainDataLoader("../data/fashion_mnist_train_vectors.csv",
                               "../data/fashion_mnist_train_labels.csv");
    auto trainData = trainDataLoader.loadAllData(1, 784);
    // Shuffle train data
    std::random_shuffle(trainData.begin(), trainData.end(), [&](int i){ return std::rand() % i; });

    // Train
    float learningRate = 0.0005;
    float momentumFactor = 0.9;
    network.train(5, 100, learningRate, momentumFactor, trainData);

    // Load test data
    DataLoader testDataLoader("../data/fashion_mnist_test_vectors.csv");
    auto testData = testDataLoader.loadAllPictures(1, 784);
    
    // Predict
    auto predictionOutputs = network.predict(testData);
    
    // Write predictions to output file
    std::cout << "Writing predictions to file: actualPredictions ... ";   
    std::ofstream actualPredictions("../actualPredictions");
    for (const auto& output : predictionOutputs)
    {
        actualPredictions << Network::findMaxIndex(output) << std::endl;
    }
    actualPredictions.close();
    std::cout << "Done" << std::endl;

    return 0;
}
