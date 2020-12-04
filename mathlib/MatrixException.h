#pragma once

#include <stdexcept>
#include <string>

namespace mathlib
{

class MatrixException : public std::runtime_error
{
    // Constructors / destructor
public:
    explicit MatrixException(const std::string& msg) :
        std::runtime_error(msg)
    {
    }
};

}
