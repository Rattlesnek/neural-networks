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
    mathlib::Matrix totalBiasUpdate; // column vector

    mathlib::Matrix previousWeightUpdate;
    mathlib::Matrix previousBiasUpdate;

    mathlib::Matrix RMSpropWeight;
    mathlib::Matrix RMSpropBias;

    // Constructors / destructor
public:
    Dense(std::string name,
        std::shared_ptr<ILayer> previousLayer,
        int numOfNeurons,       
        int PicDataSize = 1);

    // Methods
public:

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) const override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& input, const mathlib::Matrix& gradient) override;  

    virtual void updateWeights(float alpha, float momentumCoeficient) override;

    // Methods
private:
    void initializeWeights();

};

}