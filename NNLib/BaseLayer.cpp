#include "BaseLayer.h"
#include <memory>
#include "LayerException.h"

using namespace nnlib;

BaseLayer::BaseLayer(std::string name, 
    LayerType type,
    int outputHeight, 
    int outputWidth)
    :
    name(std::move(name)),
    type(type),
    outputHeight(outputHeight),
    outputWidth(outputWidth)
{
    if (outputHeight <= 0 || outputWidth <= 0)
    {
        throw LayerException(this->name + " - BaseLayer Constructor - wrong output size.");
    }
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
    return outputHeight;
}

int BaseLayer::getOutputWidth() const noexcept
{
    return outputWidth;
}