#pragma once

#include <stdexcept>
#include <string>

namespace nnlib
{

class LayerException : public std::runtime_error
{
    // Constructors / destructor
public:
    explicit LayerException(const std::string& msg) :
        std::runtime_error(msg)
    {
    }
};

}
