#pragma once

#include <stdexcept>
#include <string>

namespace dataload
{

class EOFException : public std::runtime_error
{
    // Constructors / destructor
public:
    explicit EOFException(const std::string& msg) :
        std::runtime_error(msg)
    {
    }
};

}