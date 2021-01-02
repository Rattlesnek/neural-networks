#include <numeric>
#include <cmath>
#include <cfloat>
#include "ErrorFunc.h"


using namespace mathlib;


Matrix ErrorFunc::softMax(Matrix input_copy)
{
    Matrix output(input_copy.getRows(), input_copy.getCols());
    for (int row = 0; row < input_copy.getRows(); row++ )
    {
    
        float max = -FLT_MAX;
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            if (max < input_copy(row, col))
            {
                max = input_copy(row, col);
            }
        }
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            input_copy(row, col) = input_copy(row, col) - max;
        }
        float smSum(0.0);
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            smSum = smSum + std::exp(input_copy(row, col));
        }
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            output(row, col) = (std::exp(input_copy(row, col)))/smSum;
        }    
    }
    // std::cout << " SOFTMAX FUNCTION::" << std::endl;
    // std::cout << "====================================="<< std::endl;
    // std::cout << "input_copy matrix:" << std::endl;
    // std::cout << input_copy;
    // std::cout << "input_copy matrix:" << std::endl;
    // std::cout << input_copy;
    // std::cout << "output matrix:" << std::endl;
    // std::cout << output;
    // std::cout << "======================================="<< std::endl;
    return output;
}

// jeden column 
Matrix ErrorFunc::softmaxCrossentropyWithLogits(Matrix input_copy, const std::vector<int>& label)
{
    
    Matrix output(input_copy.getRows(), 1);
    float max = 0.f;
    for (int row = 0; row < input_copy.getRows(); row++ )
    {
        float max = -FLT_MAX;
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            if (max < input_copy(row, col))
            {
                max = input_copy(row, col);
            }
        }
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            input_copy(row, col) = input_copy(row, col) - max;
        }
        float smSum(0.0);
        for (int col = 0; col < input_copy.getCols(); col++)
        {
            smSum = smSum + std::exp(input_copy(row, col));
        }
        output(row, 0) = std::log(smSum);
    }
    for (int row = 0; row < input_copy.getRows(); row++ )
    {
            output(row, 0) -= input_copy(row, label[row]);
    }
    
    return output;
}

//matica normalna size == input_copy.size()
Matrix ErrorFunc::gradSoftmaxCrossentropyWithLogits(const Matrix& input, const std::vector<int>& label)
{
    Matrix softmax = softMax(input);
    std::vector<int> cols;
    std::vector<int> rows;
    int counter = 0;
    for (int row = 0; row < input.getRows(); row++ )
    {
        softmax(row, label[row]) -= 1;
    }
    
    return softmax;

}

