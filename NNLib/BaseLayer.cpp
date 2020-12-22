#include "BaseLayer.h"
#include <memory>
#include "LayerException.h"

using namespace nnlib;
using namespace mathlib;

BaseLayer::BaseLayer(std::string name, 
    int outputHeight, 
    int outputWidth)
    :
    name(std::move(name)),
    output(outputHeight, outputWidth)
{
}

const std::string& BaseLayer::getName() const noexcept
{
    return name;
}

int BaseLayer::getOutputHeight() const noexcept
{
    return output.getRows();
}

int BaseLayer::getOutputWidth() const noexcept
{
    return output.getCols();
}

const Matrix& BaseLayer::getLastOutput() const noexcept
{
    return output;
}
