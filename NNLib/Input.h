#pragma once
#include <string>

#include "BaseLayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class Input : public BaseLayer
{
    // Constructors / destructor
public:
    Input(std::string name,
        int IOheight, 
        int IOwidth);

    // Methods
public: 
    virtual mathlib::Matrix forward(const mathlib::Matrix& input) const override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& input, const mathlib::Matrix& errorNeuronGradient) override;

};

}