#include "Dense.h"
#include "LayerException.h"
#include <random>

using namespace nnlib;
using namespace mathlib;
using namespace mathlib::activation;

Dense::Dense(std::string name,
    std::shared_ptr<ILayer> previousLayer,
    int numOfNeurons, 
    int PicDataSize)
    :
    BaseLayer(std::move(name), PicDataSize, numOfNeurons), 
    previousLayer(previousLayer)
{
    // Dummy constructor
    // TODO initialization of weights
    if (previousLayer == nullptr) 
    {
        throw LayerException(this->name + " - Constructor - nullptr passed.");
    }
    if (previousLayer->getOutputHeight() != PicDataSize)
    {
        throw LayerException(this->name + " - Constructor - could not be connected to previous layer. Size mismatch.");
    }

    // this is the matrix of weights for all neurons in the layer
    // each row of matrix is a vector of weights of a single neuron
    // this matrix of weigths will be multiplied with the input vector from the left like this: weights * input -- hence the size of the matrix
    weights.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    totalWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    previousWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());

    biases.setDimensions(1, this->getOutputWidth());
    totalBiasesUpdate.setDimensions(1, this->getOutputWidth());
    previousBiasesUpdate.setDimensions(1, this->getOutputWidth());

    initializeWeightsAndBiases();
}

Matrix Dense::forward(const Matrix& input) const
{
    //dummy implementation
    if (! input.isRowVector())
    {
        throw LayerException(name + " - Forward: input of wrong dimensions!");
    }

    // 1. Step
    return input * weights + biases;
}

Matrix Dense::backward(const Matrix& input, const Matrix& gradient)
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
    Matrix singleWeightUpdate = input.T() * gradient;
    Matrix singleBiasUpdate = gradient; // ???

    // 4. Step TODO
    #pragma omp critical
    {
        totalWeightUpdate = totalWeightUpdate + singleWeightUpdate;
        totalBiasesUpdate = totalBiasesUpdate + singleBiasUpdate;
    }
    return nextGradient.T();
}

void Dense::updateWeights(float alpha, float momentumCoeficient)
{
    // Apply learning rate
    auto multiplyByAlpha = [&](float x) -> float { return alpha * x; };
    totalWeightUpdate.applyFunc(multiplyByAlpha);
    totalBiasesUpdate.applyFunc(multiplyByAlpha);

    // Add momentum
    auto multiplyByMomentumCoef = [&](float x) -> float { return momentumCoeficient * x; };
    previousWeightUpdate.applyFunc(multiplyByMomentumCoef);
    previousBiasesUpdate.applyFunc(multiplyByMomentumCoef);
    totalWeightUpdate = totalWeightUpdate + previousWeightUpdate;
    totalBiasesUpdate = totalBiasesUpdate + previousBiasesUpdate;

    // Learn
    weights = weights - totalWeightUpdate;
    biases = biases - totalBiasesUpdate;

    // Save previous updates
    previousWeightUpdate = totalWeightUpdate;
    previousBiasesUpdate = totalBiasesUpdate;

    // Clear matrices
    totalWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    totalBiasesUpdate.setDimensions(1, this->getOutputWidth());
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
