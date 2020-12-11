#pragma once

#include <stdexcept>
#include <string>

namespace dataload
{

class DataLoadInternalException : public std::runtime_error
{
    // Constructors / destructor
public:
    explicit DataLoadInternalException(const std::string& msg) :
        std::runtime_error(msg)
    {
    }
};

}