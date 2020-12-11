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
        mat.applyFunc(relu);
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        mat.applyFunc(reluDerivative);
        return mat;
    }

public:
    static float relu(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    static float reluDerivative(float x)
    {
        return (x > 0.f) ? 1.f : 0.f;
    }
};

}