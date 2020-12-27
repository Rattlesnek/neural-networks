#include <iostream>
#include <memory>
#include <algorithm>
#include <string>
#include <omp.h>
#include <unistd.h>

#include "MathLib.hpp"
#include "DataLoad.hpp"
#include "NNLib.hpp"

using namespace mathlib;
using namespace mathlib::activation;
using namespace dataload;
using namespace nnlib;


int main(int argc, char *argv[])
{  
    bool executeTraining = false;
    if (argc == 2 && argv[1] == std::string("-t"))
    {
        executeTraining = true;
    }
    
    

    if (!executeTraining)
    {
        return 0;
    }

    

    return 0;
}
