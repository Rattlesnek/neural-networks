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
    if (type != LayerType::HiddenLayer && type != LayerType::OutputLayer)
    {
        throw LayerException(this->name + " - Constructor - incorrect layer type.");
    }
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
    weights.setDimensions(this->getOutputHeight(), previousLayer->getOutputHeight());
    totalWeightUpdate.setDimensions(this->getOutputHeight(), previousLayer->getOutputHeight());

    biases.setDimensions(this->getOutputHeight(), 1);
    totalBiasesUpdate.setDimensions(this->getOutputHeight(), 1);

    initializeWeightsAndBiases();

    neuronState.setDimensions(this->getOutputHeight(), 1);
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
    output = activation->call(neuronState);
    return output;
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
    // calculate single weight and bias update
    Matrix singleWeightUpdate = columnVector * previousLayer->getLastOutput().T();
    Matrix singleBiasUpdate = columnVector; // ???

    // 4. Step
    totalWeightUpdate = totalWeightUpdate + singleWeightUpdate;
    totalBiasesUpdate = totalBiasesUpdate + singleBiasUpdate;

    // TEMPORARY -- update now
    const float alpha = 0.5f;
    auto multiplyByAlpha = [&](float x) -> float { return alpha * x; };
    totalWeightUpdate.applyFunc(multiplyByAlpha);
    weights = weights - totalWeightUpdate;
    totalBiasesUpdate.applyFunc(multiplyByAlpha);
    biases = biases - totalBiasesUpdate;
    // TEMPORARY

    return nextErrorNeuronGradient.T();
}

void Dense::initializeWeightsAndBiases()
{
    // TODO
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, 1.f);

    auto randomInitialization = [&](float x) -> float { return distribution(generator); };
    weights.applyFunc(randomInitialization);
    biases.applyFunc(randomInitialization);
}
