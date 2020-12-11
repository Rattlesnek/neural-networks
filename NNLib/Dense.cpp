#include "Dense.h"
#include "LayerException.h"

using namespace nnlib;
using namespace mathlib;
using namespace mathlib::activation;

Dense::Dense(std::string name,
    int numOfNeurons, 
    std::shared_ptr<ILayer> previousLayer, 
    std::shared_ptr<IActivation> activation) 
    :
    name(std::move(name)),
    outputHeight(numOfNeurons),
    previousLayer(previousLayer),
    activation(activation)
{
    // Dummy constructor
    // TODO initialization of weights

    if (previousLayer->getOutputWidth() != 1)
    {
        throw LayerException("Layer '" + name + "' could not be connected to previous layer. Size mismatch.");
    }

    // this is the matrix of weights for all neurons in the layer
    // each row of matrix is a vector of weights of a single neuron
    // this matrix of weigths will be multiplied with the input vector from the left like this: weights * input -- hence the size of the matrix
    weights.setDimensions(outputHeight, previousLayer->getOutputHeight());

    biases.setDimensions(outputHeight, 1);
}

const std::string& Dense::getName() const
{
    return name;
}

int Dense::getOutputHeight() const
{
    return outputHeight;
}

int Dense::getOutputWidth() const
{
    return outputWidth;
}

Matrix Dense::forward(const Matrix& input) const
{
    return activation->call(weights * input + biases);
}