#include "ActivationFunc.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

using namespace mathlib;

double ActivationFunc::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x)); 
}

double ActivationFunc::sigmoidPrime(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

double ActivationFunc::ReLU(double x)
{
    return (x > 0.0) ? x : 0.0;
}

double ActivationFunc::ReLUPrime(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

std::vector<double> ActivationFunc::softmax(std::vector<double> vec)
{
    std::for_each(vec.begin(), vec.end(), [](double& x){ x = std::exp(x); });
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    std::for_each(vec.begin(), vec.end(), [&](double& x){ x = x / sum; });
    return vec;
}