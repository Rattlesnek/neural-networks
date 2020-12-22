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

Matrix ErrorFunc::SoftMax(const Matrix& input, const Matrix& label)
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

float ErrorFunc::SoftmaxCrossentropyWithLogits(const Matrix& input, const Matrix& label)
{
    float smSum(0.0);
    for (auto z : input.getVector())
    {
        smSum = smSum + std::exp(z);
    }
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

Matrix ErrorFunc::GradSoftmaxCrossEntropyWithLogits(const Matrix& input, const Matrix& label)
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

