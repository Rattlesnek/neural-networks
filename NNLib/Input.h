#pragma once
#include <memory>
#include <string>

#include "BaseLayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class Input : public BaseLayer
{
    // Fields
private:
    mathlib::Matrix neuronOutput;

    // Constructors / destructor
public:
    Input(std::string name,
        int IOheight, 
        int IOwidth);

    // Methods
public: 
    virtual const mathlib::Matrix& getNeuronOutput() const noexcept override;

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& errorNeuronGradient) override;

};

}