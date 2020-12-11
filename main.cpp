#include <iostream>
#include "MathLib.hpp"
#include "DataLoad.hpp"

#include <memory>

using namespace mathlib;
using namespace mathlib::activation;

int main()
{   
    Matrix mat1(2,3,{1,2,3,4,5,6});
    Matrix mat2(3,2,{7,8,9,10,11,12});
    Matrix outcome = mat1 * mat2;
    std::cout << mat1;
    std::cout << mat2;
    std::cout << outcome;
    Matrix Omat(3,3);
    for (int i = 0; i < 9; i++){
        Omat[i] = i;
    }
    Matrix Tmat = Omat.T();
    std::cout << "Original" << std::endl;
    std::cout << Omat;
    std::cout << "TRANSPOSED" << std::endl;
    std::cout << Tmat;

    DataLoader dl("../data/fashion_mnist_train_vectors.csv", "../data/fashion_mnist_train_labels.csv");
    std::vector<PicData> pd = dl.DataLoader::loadAllData(28,28);
    std::cout << pd[1].getMat() << '\n';
    std::cout << pd[1].getLabel() << '\n';

    return 0;
}