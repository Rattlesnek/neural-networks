#pragma once
#include "IActivation.h"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace mathlib::activation
{
class LogSoftMax
{
public:
    static Matrix LogSoftMaxWithCrossEntropy(const Matrix& input, const Matrix& label)
    {
        float smSum(0.0);
        for (auto z : input.getVector())
        {
            smSum = smSum + std::exp(z);
        }
        Matrix output(1, input.getVector().size());
        for (int i = 0; i < input.getVector().size(); i++ )
        {
            output[i] = (float)-input[i] + std::log(smSum);
        }
        return output;
    }
    static Matrix SoftMax(const Matrix& input, const Matrix& label)
    {
        float smSum(0.0);
        for (auto z : input.getVector())
        {
            smSum = smSum + std::exp(z);
        }
        Matrix output(1, input.getVector().size());
        for (int i = 0; i < input.getVector().size(); i++ )
        {
            output[i] = (std::exp(input[i]))/smSum;
        }
        return output;
    }
    static float SoftmaxCrossentropyWithLogits(const Matrix& input, const Matrix& label)
    {
        float smSum(0.0);
        for (auto z : input.getVector())
        {
            smSum = smSum + std::exp(z);
        }
        float xenotropy(0.0);
        int i_correct;
        for (int i = 0; i < input.getVector().size(); i++ )
        {
            if (label[i] == 1)
            {
                i_correct = i;
            }
        }
        
        return -input[i_correct] + std::log(smSum);
    }
    static Matrix GradSoftmaxCrossEntropyWithLogits(const Matrix& input, const Matrix& label)
    {
        Matrix softmax = SoftMax(input,label);
        int i_correct;
        for (int i = 0; i < input.getVector().size(); i++ )
        {
            if (label[i] == 1)
            {
                i_correct = i;
            }
        }
        softmax[i_correct] = softmax[i_correct] - 1;
        return softmax;

    }
};
}