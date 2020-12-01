#pragma once

#include <stdexcept>
#include <string>

namespace mathlib{

class MatrixWrongDimensionsException : public std::runtime_error
{
    public:
        explicit MatrixWrongDimensionsException(const std::string& msg) :
        std::runtime_error(msg)
        {
        }
};
}
