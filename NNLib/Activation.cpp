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
    outputHeight = previousLayer->getOutputHeight();
    outputWidth = previousLayer->getOutputWidth();
}

Matrix Activation::forward(const Matrix& input) const
{
    if (input.getRows() != outputHeight || input.getCols() != outputWidth)
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }
    
    return activation->call(input);
}

Matrix Activation::backward(const Matrix& input, const Matrix& gradient)
{
    return Matrix::arrayMult(gradient, activation->callDerivative(input));
}