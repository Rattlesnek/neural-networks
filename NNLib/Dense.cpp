#include "Dense.h"
#include "LayerException.h"
#include <random>

using namespace nnlib;
using namespace mathlib;
using namespace mathlib::activation;

Dense::Dense(std::string name,
    LayerType type,
    int numOfNeurons, 
    std::shared_ptr<ILayer> previousLayer, 
    std::shared_ptr<IActivation> activation) 
    :
    BaseLayer(std::move(name), type, numOfNeurons, 1), // output is by default a column vector
    previousLayer(previousLayer),
    activation(activation)
{
    // Dummy constructor
    // TODO initialization of weights
    if (previousLayer == nullptr || activation == nullptr)
    {
        throw LayerException(this->name + " - Constructor - nullptr passed.");
    }
    if (previousLayer->getOutputWidth() != 1)
    {
        throw LayerException(this->name + " - Constructor - could not be connected to previous layer. Size mismatch.");
    }

    // this is the matrix of weights for all neurons in the layer
    // each row of matrix is a vector of weights of a single neuron
    // this matrix of weigths will be multiplied with the input vector from the left like this: weights * input -- hence the size of the matrix
    weights.setDimensions(outputHeight, previousLayer->getOutputHeight());
    initializeWeights();
    totalWeightUpdate.setDimensions(outputHeight, previousLayer->getOutputHeight());

    biases.setDimensions(outputHeight, 1);

    neuronState.setDimensions(outputHeight, 1);
    neuronOutput.setDimensions(outputHeight, 1);
}

const Matrix& Dense::getNeuronOutput() const
{
    return neuronOutput;
}

Matrix Dense::forward(const Matrix& input)
{
    // dummy implementation
    if (! input.isColumnVector())
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }

    // 1. Step
    neuronState = weights * input + biases;
    neuronOutput = activation->call(neuronState);
    return neuronOutput;
}

Matrix Dense::backward(const Matrix& errorNeuronGradient)
{
    // dummy implementation
    if (! errorNeuronGradient.isColumnVector())
    {
        throw LayerException(name + " - Backward: errorNeuronGradient of wrong dimensions!");
    }

    // 2. Step
    // calculate nextErrorNeuronGradient
    Matrix columnVector = Matrix::arrayMult(errorNeuronGradient, activation->callDerivative(neuronState));
    Matrix rowVector = columnVector.T();
    Matrix nextErrorNeuronGradient = rowVector * weights;

    // 3. Step
    // calculate single weight update
    Matrix otherColumnVector = Matrix::arrayMult(errorNeuronGradient, activation->callDerivative(neuronState)); 
    Matrix singleWeightUpdate = otherColumnVector * previousLayer->getNeuronOutput().T();

    // 4. Step
    totalWeightUpdate = totalWeightUpdate + singleWeightUpdate;

    return nextErrorNeuronGradient;
}

void Dense::initializeWeights()
{
    // TODO
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, 1.f);
    weights.applyFunc([&](float x) -> float { return distribution(generator); });
}
