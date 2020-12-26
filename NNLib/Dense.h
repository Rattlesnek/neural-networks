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

    mathlib::Matrix weights; // matrix
    mathlib::Matrix biases; // column vector

    mathlib::Matrix totalWeightUpdate; // matrix
    mathlib::Matrix totalBiasesUpdate; // column vector

    // Constructors / destructor
public:
    Dense(std::string name,
        std::shared_ptr<ILayer> previousLayer,
        int numOfNeurons,       
        int batchSize = 1);

    

    // Methods
public:

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& gradient) override;  

    virtual void updateWeights(int it) override;

    // Methods
private:
    void initializeWeightsAndBiases();

};

}