#pragma once
#include <vector>

namespace mathlib
{

class ErrorFunc
{
    // Constructors / destructor
public:
    ErrorFunc() = delete;

    // Methods
public:

    static double categoricalCrossentropy(const std::vector<double>& outputs, const std::vector<double>& labels);

};

}