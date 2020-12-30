#include "TrainUtils.h"

using namespace nnlib;

float TrainUtils::powerSchedulingLR(float LR, int numOfBatchesInEpoch, int currentBatch)
{
    return LR / (1 + ((float)currentBatch / (float)numOfBatchesInEpoch));
}

float TrainUtils::exponentialScheduling(float LR, int numOfBatchesInEpoch, int currentBatch)
{
    return LR * std::pow(0.1f, (float)currentBatch /(float) numOfBatchesInEpoch);
}

float TrainUtils::piecewiseConstantScheduling(float LR, int numOfBatchesInEpoch, int currentBatch)
{
    auto multiplier = ((numOfBatchesInEpoch / 100) / currentBatch);
    return LR;
}

float TrainUtils::oneCycleScheduling(float LR, int maxBatches, int currentBatch)
{
    int tmp = maxBatches / 4;
    if (currentBatch < tmp)
    {
        return LR;
    }
    else
    {   
        return LR * (1.f - ((float)(currentBatch - tmp) / (float)(maxBatches - tmp + 1)));
    }
}