#include "BaseLayer.h"
#include <memory>
#include "LayerException.h"

using namespace nnlib;
using namespace mathlib;

BaseLayer::BaseLayer(std::string name, 
    LayerType type,
    int outputHeight, 
    int outputWidth)
    :
    name(std::move(name)),
    type(type),
    output(outputHeight, outputWidth)
{
}

const std::string& BaseLayer::getName() const noexcept
{
    return name;
}

const LayerType& BaseLayer::getType() const noexcept
{
    return type;
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
