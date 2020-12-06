#pragma once
#include <vector>

namespace mathlib
{

class ActivationFunc
{
    // Constructors / destructor
public:
    ActivationFunc() = delete;

    // Methods
public:

    static double sigmoid(double x);

    static double sigmoidPrime(double x);

    static double ReLU(double x);

    static double ReLUPrime(double x);

    static std::vector<double> softmax(std::vector<double> vec);

};

}