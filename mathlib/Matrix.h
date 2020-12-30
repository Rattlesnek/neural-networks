#pragma once

#include <vector>
#include <functional>
#include <iostream>

namespace mathlib
{

class Matrix
{
    // Fields
private:    
    int rows;
    int cols;
    std::vector<float> mat;
    
    // Constructors / destructor
public:
    Matrix() = default;

    Matrix(int rows, int cols);

    Matrix(int rows, int cols, std::vector<float> vec);

    // Methods 
public:
    const int getRows() const noexcept;

    const int getCols() const noexcept;

    const std::vector<float>& getVector() const noexcept;

    bool isColumnVector() const noexcept;

    bool isRowVector() const noexcept;

    void setDimensions(int rows, int cols);

    void print() const;

    void applyFunc(std::function<float(float)> func);

    Matrix T() const;

    float sum() const;

    // Operators
public:
    float& operator()(int row, int col);

    float operator()(int row, int col) const;

    Matrix operator+(const Matrix& m2) const;

    Matrix operator-(const Matrix& m2) const;

    Matrix operator*(const Matrix& m2) const;

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& m);

    // Static methods
public:
    static Matrix arrayMult(const Matrix& m1, const Matrix& m2);

};

float round(float i);

std::ostream& operator<<(std::ostream& stream, const Matrix& m);

}