#include "ErrorFunc.h"
#include <numeric>
#include <cmath>

using namespace mathlib;

double ErrorFunc::categoricalCrossentropy(const std::vector<double>& outputs, const std::vector<double>& labels)
{
    // categorical crossentropy = - sum(labels * log(ouputs))
    std::vector<double> multLabels;
    for (int i = 0; i < labels.size(); i++)
    {
        multLabels.emplace_back(labels[i] * std::log(outputs[i]));
    }
    return - std::accumulate(multLabels.begin(), multLabels.end(), 0.0);  
}

