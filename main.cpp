#include <iostream>
#include "MathLib.hpp"
#include "DataLoad.hpp"
#include "NNLib.hpp"

#include <memory>

using namespace mathlib;
using namespace mathlib::activation;
using namespace dataload;
using namespace nnlib;

int main()
{  
    std::shared_ptr<ILayer> input = std::make_shared<Input>("Input_name", 5, 1);
    std::cout << input->getName() << std::endl;
    std::cout << input->getType() << std::endl;
    std::cout << input->getOutputHeight() << std::endl;
    std::cout << input->getOutputWidth() << std::endl;
    std::cout << input->getNeuronOutput() << std::endl;

    Dense dense("Dense_name", LayerType::HiddenLayer, 10, input, std::make_shared<ReLU>());
    std::cout << dense.getName() << std::endl;
    std::cout << dense.getType() << std::endl;
    std::cout << dense.getOutputHeight() << std::endl;
    std::cout << dense.getOutputWidth() << std::endl;
    std::cout << dense.getNeuronOutput() << std::endl;

    return 0;
}