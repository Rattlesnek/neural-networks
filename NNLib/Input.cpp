#include "Input.h"
#include <memory>
#include "LayerException.h"

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
    if (input.getRows() != output.getRows() || input.getCols() != output.getCols())
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }

    output = input;
    return output;
}

Matrix Input::backward(const Matrix& errorNeuronGradient)
{
    return errorNeuronGradient;
}
