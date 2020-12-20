#pragma once
#include <memory>
#include <string>

#include "BaseLayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class Dense : public BaseLayer
{
    // Fields
private:
    std::shared_ptr<ILayer> previousLayer;
    // std::shared_ptr<ILayer> nextLayer;

    mathlib::Matrix weights; // matrix
    mathlib::Matrix biases; // column vector

    mathlib::Matrix neuronState; // column vector
    mathlib::Matrix neuronOutput; // column vector

    mathlib::Matrix totalWeightUpdate; // matrix

    std::shared_ptr<mathlib::activation::IActivation> activation;


    // Constructors / destructor
public:
    Dense(std::string name,
        LayerType type,
        int numOfNeurons, // numOfNeurons = outputHeight * outputWidth
        std::shared_ptr<ILayer> previousLayer, 
        std::shared_ptr<mathlib::activation::IActivation> activation);

    // Methods
public:

    virtual const mathlib::Matrix& getNeuronOutput() const override;

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& errorNeuronGradient) override;  

    // Methods
private:
    void initializeWeights();

};

}