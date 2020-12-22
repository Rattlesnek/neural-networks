#include "Input.h"
#include <memory>

using namespace nnlib;
using namespace mathlib;

Input::Input(std::string name,
    int IOheight, 
    int IOwidth)
    :
    BaseLayer(std::move(name), IOheight, IOwidth)
{
}

Matrix Input::forward(const Matrix& input)
{
    output = input;
    return output;
}

Matrix Input::backward(const Matrix& errorNeuronGradient)
{
    return errorNeuronGradient;
}
