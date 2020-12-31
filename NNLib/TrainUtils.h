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
    static float powerSchedulingLR(float LR, int numOfBatchesInEpoch, int currentBatch);
    
    static float exponentialScheduling(float LR, int numOfBatchesInEpoch, int currentBatch);
    
    static float piecewiseScheduling(float LR, int maxBatches, int currentBatch);

    static float oneCycleScheduling(float LR, int maxBatches, int currentBatch);
    
};
}