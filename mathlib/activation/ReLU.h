#pragma once
#include "IActivation.h"

namespace mathlib::activation
{

class ReLU : public IActivation
{
    // Methods
public:
    virtual Matrix call(Matrix mat) override
    {
        for (auto& elem : mat)
        {
            elem = relu(elem);
        }
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        for (auto& elem : mat)
        {
            elem = reluDerivative(elem);
        }
        return mat;
    }

private:
    double relu(double x)
    {
        return (x > 0.0) ? x : 0.0;
    }

    double reluDerivative(double x)
    {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};

}