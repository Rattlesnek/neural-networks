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
    RMSpropWeight.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());

    biases.setDimensions(1, this->getOutputWidth());
    totalBiasUpdate.setDimensions(1, this->getOutputWidth());
    previousBiasUpdate.setDimensions(1, this->getOutputWidth());
    RMSpropBias.setDimensions(1, this->getOutputWidth());

    initializeWeights();
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
        totalBiasUpdate = totalBiasUpdate + singleBiasUpdate;
    }
    return nextGradient.T();
}

void Dense::updateWeights(float alpha, float momentumFactor)
{   
    // RMSProp
    const float decay = 0.9f;
    
    auto square = [&](float x) -> float { return x * x; };
    auto decayFunc = [&](float x) -> float { return decay * x; };
    auto decayFuncInverse = [&](float x) -> float { return (1.f - decay) * x; };

    RMSpropWeight = RMSpropWeight.func(decayFunc) + totalWeightUpdate.func(square).func(decayFuncInverse);
    RMSpropBias = RMSpropBias.func(decayFunc) + totalBiasUpdate.func(square).func(decayFuncInverse);

    // Apply RMSProp
    auto fraction = [&](float x) -> float { return (float) alpha / std::sqrt(x + 1.e-8f); };
    totalWeightUpdate = Matrix::arrayMult(RMSpropWeight.func(fraction), totalWeightUpdate);
    totalBiasUpdate = Matrix::arrayMult(RMSpropBias.func(fraction), totalBiasUpdate);

    // Add momentum
    auto multiplyByMomentumFactor = [&](float x) -> float { return momentumFactor * x; };
    totalWeightUpdate = totalWeightUpdate + previousWeightUpdate.func(multiplyByMomentumFactor);
    totalBiasUpdate = totalBiasUpdate + previousBiasUpdate.func(multiplyByMomentumFactor);

    // Learn
    weights = weights - totalWeightUpdate;
    biases = biases - totalBiasUpdate;

    // Save previous updates
    previousWeightUpdate = totalWeightUpdate;
    previousBiasUpdate = totalBiasUpdate;

    // Clear matrices
    totalWeightUpdate.setDimensions(previousLayer->getOutputWidth(), this->getOutputWidth());
    totalBiasUpdate.setDimensions(1, this->getOutputWidth());
}

void Dense::initializeWeights()
{
    // Kaiming initialization / He initialization
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, std::sqrt(2.f / (float)previousLayer->getOutputWidth()));
    auto randomInitialization = [&](float x) -> float { auto z = distribution(generator); return z; };
    weights.applyFunc(randomInitialization);
}
