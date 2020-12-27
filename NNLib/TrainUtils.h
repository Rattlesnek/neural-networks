#pragma once

#include<cmath>
namespace nnlib
{

class TrainUtils
{
    // Constructors / destructor
public:
    TrainUtils() = delete;

    // Methods
public:
    static float powerSchedulingLR(int LR, int epochSteps, int currSteps);
    static float exponentialScheduling(int LR, int epochSteps, int currSteps);
    static float piecewiseConstantScheduling(int LR, int epochSteps, int currSteps);
    static float oneCycleScheduling(int LR, int epochSteps, int currSteps);
};
}