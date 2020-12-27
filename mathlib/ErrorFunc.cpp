#include <numeric>
#include <cmath>
#include <cfloat>
#include "ErrorFunc.h"


using namespace mathlib;


Matrix ErrorFunc::softMax(const Matrix& input)
{
    Matrix input_copy = input;
    Matrix output(input.getRows(), input.getCols());
    for (int row = 0; row < input.getRows(); row++ )
    {
    
        float max = -FLT_MAX;
        for (int col = 0; col < input.getCols(); col++)
        {
            if (max < input_copy(row, col))
            {
                max = input_copy(row, col);
            }
        }
        for (int col = 0; col < input.getCols(); col++)
        {
            input_copy(row, col) = input_copy(row, col) - max;
        }
        float smSum(0.0);
        for (int col = 0; col < input.getCols(); col++)
        {
            smSum = smSum + std::exp(input_copy(row, col));
        }
        for (int col = 0; col < input.getCols(); col++)
        {
            output(row, col) = (std::exp(input_copy(row, col)))/smSum;
        }    
    }
    // std::cout << " SOFTMAX FUNCTION::" << std::endl;
    // std::cout << "====================================="<< std::endl;
    // std::cout << "input matrix:" << std::endl;
    // std::cout << input;
    // std::cout << "input_copy matrix:" << std::endl;
    // std::cout << input_copy;
    // std::cout << "output matrix:" << std::endl;
    // std::cout << output;
    // std::cout << "======================================="<< std::endl;
    return output;
}

// jeden column 
Matrix ErrorFunc::softmaxCrossentropyWithLogits(const Matrix& input, const std::vector<int>& label)
{
    Matrix input_copy = input;
    Matrix output(input.getRows(), 1);
    float max = 0.f;
    for (int row = 0; row < input.getRows(); row++ )
    {
        float max = -FLT_MAX;
        for (int col = 0; col < input.getCols(); col++)
        {
            if (max < input_copy(row, col))
            {
                max = input_copy(row, col);
            }
        }
        for (int col = 0; col < input.getCols(); col++)
        {
            input_copy(row, col) = input_copy(row, col) - max;
        }
        float smSum(0.0);
        for (int col = 0; col < input.getCols(); col++)
        {
            smSum = smSum + std::exp(input_copy(row, col));
        }
        output(row, 0) = std::log(smSum);
    }
    for (int row = 0; row < input.getRows(); row++ )
    {
            output(row, 0) -= input_copy(row, label[row]);
    }
    
    return output;
}

//matica normalna size == input.size()
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

