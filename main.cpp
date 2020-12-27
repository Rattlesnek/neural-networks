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
    omp_set_num_threads(4);

    DataLoader dl = DataLoader("../data/fashion_mnist_train_vectors.csv",
                               "../data/fashion_mnist_train_labels.csv");

    // Change the leftmost number here for taking data out
    std::vector<PicData> data = dl.loadNOfEach(300, 1, 784);
    std::random_shuffle(data.begin(), data.end(),[&](int i) {return std::rand() % i;} );

    auto [validationData, trainData] = PreprocessingUtils::splitDataValidTrain(0.1, data);

    network.train(2, 100, trainData);
    
    return 0;
}
