#pragma once
#include <iostream>
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
    static float powerSchedulingLR(float LR, int epochSteps, int currSteps);
    static float exponentialScheduling(float LR, int epochSteps, int currSteps);
    static float piecewiseConstantScheduling(float LR, int epochSteps, int currSteps);
    static float oneCycleScheduling(float LR, int epochSteps, int currSteps);
};
}