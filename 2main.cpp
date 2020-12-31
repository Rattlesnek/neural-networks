#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>
#include <unistd.h>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <omp.h>

#include "MathLib.hpp"
#include "DataLoad.hpp"
#include "NNLib.hpp"

using namespace mathlib;
using namespace mathlib::activation;
using namespace dataload;
using namespace nnlib;


int main(int argc, char *argv[])
{  
    int max = 100;
    for (int i = 0; i <= max; i++)
    {
        std::cout << i << ": "<<TrainUtils::piecewiseScheduling(100, max, i) << std::endl;
    }

    return 0 ;
}
