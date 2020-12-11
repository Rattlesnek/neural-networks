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
    float relu(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    float reluDerivative(float x)
    {
        return (x > 0.f) ? 1.f : 0.f;
    }
};

}