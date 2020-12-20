#include "Input.h"
#include <memory>

using namespace nnlib;
using namespace mathlib;

Input::Input(std::string name,
    int IOheight, 
    int IOwidth)
    :
    BaseLayer(std::move(name), LayerType::InputLayer, IOheight, IOwidth)
{
    neuronOutput.setDimensions(IOheight, IOwidth);
}

const Matrix& Input::getNeuronOutput() const noexcept
{
    return neuronOutput;
}

Matrix Input::forward(const Matrix& input)
{
    return input;
}

Matrix Input::backward(const Matrix& errorNeuronGradient)
{
    // TODO
    return errorNeuronGradient;
}