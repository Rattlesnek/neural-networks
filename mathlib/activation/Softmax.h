#pragma once
#include "IActivation.h"
#include <numeric>
#include <algorithm>

namespace mathlib::activation
{

class Softmax : public IActivation
{
    // Methods
public:
    virtual Matrix call(Matrix mat) override
    {
        std::for_each(mat.begin(), mat.end(), [](double& x){ x = std::exp(x); });
        double sum = mat.sum();
        std::for_each(mat.begin(), mat.end(), [&](double& x){ x = x / sum; });
        return mat;
    }

    virtual Matrix callDerivative(Matrix mat) override
    {
        // TODO not implemented
        return Matrix();
    }
};

}