#pragma once 

#include <string>
#include "Matrix.h"

namespace nnlib
{

class ILayer
{
    // Constructors / destructor
public:
    virtual ~ILayer() = default;

    // Methods
public:
    virtual const std::string& getName() const = 0;

    virtual int getOutputHeight() const = 0;

    virtual int getOutputWidth() const = 0;

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) const = 0;
    
    virtual mathlib::Matrix backward(const mathlib::Matrix& input, const mathlib::Matrix& gradient) = 0;

    virtual void updateWeights(float alpha, float momentumCoeficient) = 0;

};

}