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
float ErrorFunc::meanSquareError(const Matrix& predictions, const Matrix& labels)
{
    return 0.5;
}

Matrix ErrorFunc::softMax(const Matrix& input)
{
    Matrix output(input.getRows(), input.getCols());
    for (int row = 0; row < input.getRows(); row++ )
    {
        float smSum(0.0);
        for (int col = 0; col < input.getCols(); col++)
        {
            smSum = smSum + std::exp(input(row, col));
        }
        for (int col = 0; col < input.getCols(); col++)
        {
            output(row, col) = (std::exp(input(row, col)))/smSum;
        }
        
    }
    return output;
}

// jeden column 
Matrix ErrorFunc::softmaxCrossentropyWithLogits(const Matrix& input, const Matrix& label)
{
    Matrix output(input.getRows(), 1);
    for (int row = 0; row < input.getRows(); row++ )
    {
        float smSum(0.0);
        for (int col = 0; col < input.getCols(); col++)
        {
            smSum = smSum + std::exp(input(row, col));
        }
        output(row, 0) = std::log(smSum);
    }
    for (int row = 0; row < input.getRows(); row++ )
    {
        for (int col = 0; col < input.getCols(); col++)
        if (label(row, col) == 1)
        {
            output(row, 0) -= input(row, col);
        }
    }
    return output;
}

//matica normalna size == input.size()
Matrix ErrorFunc::gradSoftmaxCrossentropyWithLogits(const Matrix& input, const Matrix& label)
{
    Matrix softmax = softMax(input);
    std::vector<int> cols;
    std::vector<int> rows;
    int counter = 0;
    for (int row = 0; row < input.getRows(); row++ )
    {
        for (int col = 0; col < input.getCols(); col++)
        if (label(row, col) == 1)
        {
            softmax(row, col) -= 1;
        }
    }
    
    return softmax;

}

