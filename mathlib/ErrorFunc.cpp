#include "ErrorFunc.h"
#include <numeric>
#include <cmath>

using namespace mathlib;

float ErrorFunc::categoricalCrossentropy(const Matrix& predictions, const Matrix& labels)
{
    // categorical crossentropy = - sum(labels * log(ouputs))
    std::vector<float> multLabels;
    for (int i = 0; i < labels.getRows(); i++)
    {   
        for (int j = 0; j < labels.getCols(); j++)
        {
            multLabels.emplace_back(labels(i, j) * std::log(predictions(i, j)));
        }
    }
    return - std::accumulate(multLabels.begin(), multLabels.end(), 0.f);  
}

