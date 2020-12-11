#pragma once
#include "Matrix.h"

namespace mathlib::activation
{

class IActivation
{
    // Constructors / destructor
public:
    virtual ~IActivation() = default;

    // Methods
public:
    
    virtual Matrix call(Matrix mat) = 0;

    virtual Matrix callDerivative(Matrix mat) = 0;

};

}