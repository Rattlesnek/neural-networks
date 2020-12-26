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
    BaseLayer(std::move(name), batchSize, numOfNeurons), 
    previousLayer(previousLayer)
{
    // Dummy constructor
    // TODO initialization of weights
    if (previousLayer == nullptr) 
    {
        throw LayerException(this->name + " - Constructor - nullptr passed.");
    }
    if (previousLayer->getOutputHeight() != batchSize)
    {
        throw LayerException(this->name + " - Constructor - could not be connected to previous layer. Size mismatch.");
    }

    // this is the matrix of weights for all neurons in the layer
    // each row of matrix is a vector of weights of a single neuron
    // this matrix of weigths will be multiplied with the input vector from the left like this: weights * input -- hence the size of the matrix
    weights.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    totalWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());

    biases.setDimensions(1, this->getOutputWidth());
    totalBiasesUpdate.setDimensions(1, this->getOutputWidth());

    initializeWeightsAndBiases();
}

Matrix Dense::forward(const Matrix& input)
{
    //dummy implementation
    if (! input.isRowVector())
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }

    // 1. Step
    output = input * weights + biases;
    return output;
}

Matrix Dense::backward(const Matrix& gradient)
{
    //dummy implementation
    if (! gradient.isRowVector())
    {
        throw LayerException(name + " - Backward: gradient of wrong dimensions!");
    }

    // 2. Step
    // calculate nextGradient
    Matrix nextGradient = weights * gradient.T();

    // 3. Step
    // calculate single weight and bias update
    Matrix singleWeightUpdate = previousLayer->getLastOutput().T() * gradient;
    Matrix singleBiasUpdate = gradient; // ???

    // 4. Step TODO
    totalWeightUpdate = totalWeightUpdate + singleWeightUpdate;
    totalBiasesUpdate = totalBiasesUpdate + singleBiasUpdate;

    return nextGradient.T();
}

void Dense::updateWeights(int it)
{
    // TEMPORARY -- update now
    const float alpha = 0.1f * 1.f/(1.f + (it / 100));
    auto multiplyByAlpha = [&](float x) -> float { return alpha * x; };
    totalWeightUpdate.applyFunc(multiplyByAlpha);
    weights = weights - totalWeightUpdate;
    totalBiasesUpdate.applyFunc(multiplyByAlpha);
    biases = biases - totalBiasesUpdate;

    totalWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    totalBiasesUpdate.setDimensions(1, this->getOutputWidth());
    // TEMPORARY
}

void Dense::initializeWeightsAndBiases()
{
    // TODO
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, 1.f);
    auto randomInitialization = [&](float x) -> float { auto z = distribution(generator) * std::sqrt(1.f/ previousLayer->getOutputWidth()); return z ; };
    weights.applyFunc(randomInitialization);
    biases.applyFunc(randomInitialization);
}
