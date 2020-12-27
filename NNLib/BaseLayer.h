#pragma once
#include <string>

#include "ILayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class BaseLayer : public ILayer
{
    // Fields
protected:
    std::string name;
    int outputHeight;
    int outputWidth;

    // Constructor / destructor
public:
    BaseLayer(std::string name,
        int outputHeight, 
        int outputWidth);

    // Methods
public:
    virtual const std::string& getName() const noexcept override;

    virtual int getOutputHeight() const noexcept override;

    virtual int getOutputWidth() const noexcept override;

    virtual void updateWeights(int it) override;

};

}