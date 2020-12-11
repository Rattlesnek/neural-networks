#pragma once
#include "IActivation.h"
#include <cmath>

namespace mathlib::activation
{

class Sigmoid : public IActivation
{
    // Methods
public:
    virtual Matrix call(Matrix mat) override
    {
        mat.applyFunc(sigmoid);
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        mat.applyFunc(sigmoidDerivative);
        return mat;
    }

public:
    static float sigmoid(float x)
    {
        return 1.f / (1.f + std::exp(-x)); 
    }

    static float sigmoidDerivative(float x)
    {
        return sigmoid(x) * (1.f - sigmoid(x));
    }
};

}