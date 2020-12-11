#pragma once
#include <memory>
#include <string>

#include "ILayer.h"
#include "MathLib.hpp"

namespace nnlib
{

class Dense : ILayer
{
    // Fields
private:
    std::string name;
    // numOfNeurons = outputHeight * outputWidth
    int outputHeight; 
    int outputWidth = 1; // output is by default a column vector (mathlib::Matrix)

    std::shared_ptr<ILayer> previousLayer;
    // std::shared_ptr<ILayer> nextLayer;

    mathlib::Matrix weights;
    mathlib::Matrix biases;

    std::shared_ptr<mathlib::activation::IActivation> activation;


    // Constructors / destructor
public:
    Dense(std::string name,
        int numOfNeurons,
        std::shared_ptr<ILayer> previousLayer, 
        std::shared_ptr<mathlib::activation::IActivation> activation);

    // Methods
public:

    virtual const std::string& getName() const override;

    virtual int getOutputHeight() const override;

    virtual int getOutputWidth() const override;

    virtual mathlib::Matrix forward(const mathlib::Matrix& input) const override;



};

}