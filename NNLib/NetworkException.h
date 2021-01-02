#pragma once

#include <stdexcept>
#include <string>

namespace nnlib
{

class NetworkException : public std::runtime_error
{
    // Constructors / destructor
public:
    explicit NetworkException(const std::string& msg) :
        std::runtime_error(msg)
    {
    }
};

}
