
#include "TrainUtils.h"

using namespace nnlib;

static float powerSchedulingLR(int LR, int epochSteps, int currSteps)
{
    return 0.001f/(1 + (float)currSteps/ (float) epochSteps);
}
static float exponentialScheduling(int LR, int epochSteps, int currSteps)
{
    return 0.001f* std::pow(0.1f, (float)currSteps/(float) epochSteps);
}
static float piecewiseConstantScheduling(int LR, int epochSteps, int currSteps)
{
    auto multiplier = ((epochSteps / 100)/ currSteps);
    return 0.001f;
}
static float oneCycleScheduling(int LR, int epochSteps, int currSteps)
{
    return 0.001f;
}