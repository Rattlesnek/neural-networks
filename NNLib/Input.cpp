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

Matrix Input::forward(const Matrix& input) const
{
    if (input.getRows() != outputHeight || input.getCols() != outputWidth)
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }

    return input;
}

Matrix Input::backward(const Matrix& input, const Matrix& errorNeuronGradient)
{
    return errorNeuronGradient;
}
