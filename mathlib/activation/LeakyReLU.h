#pragma once
#include "IActivation.h"

namespace mathlib::activation
{

class LeakyReLU : public IActivation
{
    // Methods
public:
    virtual Matrix call(Matrix mat) override
    {
        mat.applyFunc(leakyRelu);
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        mat.applyFunc(leakyReluDerivative);
        return mat;
    }

public:
    static float leakyRelu(float x)
    {
        return (x > 0.f) ? x : 0.001 * x;
    }

    static float leakyReluDerivative(float x)
    {
        return (x > 0.f) ? 1.f : 0.001f;
    }
};

}