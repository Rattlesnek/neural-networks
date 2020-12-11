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

    void setDimensions(int rows, int cols);

    void print() const;

    Matrix applyFunc(std::function<float(float)> func) const;

    Matrix T() const;

    float sum() const;

    std::vector<float>::iterator begin();

    std::vector<float>::iterator end();

    // Operators
public:
    float& operator()(int row, int col);

    float operator()(int row, int col) const;

    float& operator[](int i); // added this so we can more easily generate matrices for testing
    
    float operator[](int i) const;

    Matrix operator+(const Matrix& m2) const;

    Matrix operator*(const Matrix& m2) const;

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& m);
};

std::ostream& operator<<(std::ostream& stream, const Matrix& m);

}