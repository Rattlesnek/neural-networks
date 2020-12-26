#pragma once
#include <string>
#include <memory>

#include "BaseLayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class Activation : public BaseLayer
{
    // Fields
private:
    std::shared_ptr<ILayer> previousLayer;    

    std::shared_ptr<mathlib::activation::IActivation> activation;

    // Constructors / destructor
public:
    Activation(std::string name,
        std::shared_ptr<ILayer> previousLayer, 
        std::shared_ptr<mathlib::activation::IActivation> activation);

    // Methods
public:

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) const override;

    virtual mathlib::Matrix backward(const mathlib::Matrix& input, const mathlib::Matrix& gradient) override;

};

}