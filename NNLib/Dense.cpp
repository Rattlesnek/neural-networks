#include "Dense.h"
#include "LayerException.h"
#include <random>

using namespace nnlib;
using namespace mathlib;
using namespace mathlib::activation;

Dense::Dense(std::string name,
    std::shared_ptr<ILayer> previousLayer,
    int numOfNeurons, 
    int batchSize)
    :
    BaseLayer(std::move(name), numOfNeurons, batchSize),
    previousLayer(previousLayer)
{
    // Dummy constructor
    // TODO initialization of weights
    if (previousLayer == nullptr) 
    {
        throw LayerException(this->name + " - Constructor - nullptr passed.");
    }
    if (previousLayer->getOutputWidth() != batchSize)
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
}

Matrix Dense::forward(const Matrix& input)
{
    // dummy implementation
    // if (! input.isColumnVector())
    // {
    //     throw LayerException(name + " - Forward: input of wrong dimensions!");
    // }

    // 1. Step
    output = weights * input + biases;
    return output;
}

Matrix Dense::backward(const Matrix& gradient)
{
    // dummy implementation
    // if (! gradient.isColumnVector())
    // {
    //     throw LayerException(name + " - Backward: gradient of wrong dimensions!");
    // }

    // 2. Step
    // calculate nextGradient
    Matrix rowVector = gradient.T();
    Matrix nextGradient = rowVector * weights;

    // 3. Step
    // calculate single weight and bias update
    Matrix singleWeightUpdate = gradient * previousLayer->getLastOutput().T();
    Matrix singleBiasUpdate = gradient; // ???

    // 4. Step TODO
    totalWeightUpdate = singleWeightUpdate;
    totalBiasesUpdate = singleBiasUpdate;

    // TEMPORARY -- update now
    const float alpha = 0.5f;
    auto multiplyByAlpha = [&](float x) -> float { return alpha * x; };
    totalWeightUpdate.applyFunc(multiplyByAlpha);
    weights = weights - totalWeightUpdate;
    totalBiasesUpdate.applyFunc(multiplyByAlpha);
    biases = biases - totalBiasesUpdate;
    // TEMPORARY

    return nextGradient.T();
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
