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
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x)); 
    }

    double sigmoidDerivative(double x)
    {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }
};

}