#include "Activation.h"
#include <memory>

#include "LayerException.h"

using namespace nnlib;
using namespace mathlib;

Activation::Activation(std::string name,
    std::shared_ptr<ILayer> previousLayer, 
    std::shared_ptr<mathlib::activation::IActivation> activation)
    :
    BaseLayer(std::move(name), 1, 1), // Temporary size
    previousLayer(previousLayer),
    activation(activation)
{
    if (previousLayer == nullptr || activation == nullptr)
    {
        throw LayerException(this->name + " - Constructor - nullptr passed.");
    }

    // set output size
    output.setDimensions(previousLayer->getOutputHeight(), previousLayer->getOutputWidth());
}

Matrix Activation::forward(const Matrix& input)
{
    output = activation->call(input);
    return output;
}

Matrix Activation::backward(const Matrix& gradient)
{
    return Matrix::arrayMult(gradient, activation->callDerivative(previousLayer->getLastOutput()));
}