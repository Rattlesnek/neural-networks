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
        for (auto& elem : mat)
        {
            elem = sigmoid(elem);
        }
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        for (auto& elem : mat)
        {
            elem = sigmoidDerivative(elem);
        }
        return mat;
    }

private:
    float sigmoid(float x)
    {
        return 1.f / (1.f + std::exp(-x)); 
    }

    float sigmoidDerivative(float x)
    {
        return sigmoid(x) * (1.f - sigmoid(x));
    }
};

}