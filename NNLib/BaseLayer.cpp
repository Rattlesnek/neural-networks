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
    outputHeight(outputHeight),
    outputWidth(outputWidth)
{
}

const std::string& BaseLayer::getName() const noexcept
{
    return name;
}

int BaseLayer::getOutputHeight() const noexcept
{
    return outputHeight;
}

int BaseLayer::getOutputWidth() const noexcept
{
    return outputWidth;
}

void BaseLayer::updateWeights(int it)
{
    // No Operation
}
