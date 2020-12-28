
#include "TrainUtils.h"
using namespace nnlib;

float TrainUtils::powerSchedulingLR(float LR, int epochSteps, int currSteps)
{
    return LR/(1 + ((float)currSteps/ (float) epochSteps));
}
float TrainUtils::exponentialScheduling(float LR, int epochSteps, int currSteps)
{
    return LR* std::pow(0.1f, (float)currSteps/(float) epochSteps);
}
float TrainUtils::piecewiseConstantScheduling(float LR, int epochSteps, int currSteps)
{
    auto multiplier = ((epochSteps / 100)/ currSteps);
    return LR;
}
float TrainUtils::oneCycleScheduling(float LR, int maxSteps, int currSteps)
{
    if (maxSteps / 2 > currSteps)
    {
        LR = LR * (1.f + 100.f/maxSteps);
    }
    else
    {
        LR = LR * (1.f - 100.f/maxSteps);
    }
    std::cout << LR << "<- learning rate";
    return LR;
}