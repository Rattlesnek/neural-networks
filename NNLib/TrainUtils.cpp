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

float TrainUtils::piecewiseScheduling(float LR, int maxBatches, int currentBatch)
{
    float time = (float)currentBatch/(float)maxBatches;

    float secondSector = 0.1f;
    float thirdSector = secondSector + 0.3f;
    float fourthSector = thirdSector + 0.4f;

    float secondFactor = 0.2f;
    float thirdFactor = 0.01f;
    float fourthFactor = 0.001f;

    if (time < secondSector)
    {
        return LR;
    }
    
    if (time <= thirdSector)
    {   
        float x = ((1 - secondFactor) * LR) * (((currentBatch - (maxBatches * secondSector)) / ((maxBatches * thirdSector - (maxBatches * secondSector)))));
        return LR - x;
    }

    if (time <= fourthSector)
    {

        float x = ((secondFactor - thirdFactor) *LR) * (((currentBatch - (maxBatches * thirdSector)) / ((maxBatches * fourthSector - (maxBatches * thirdSector)))));
        return secondFactor * LR - x;
    }

    if (time <= 1)
    {

        float x = ((thirdFactor - fourthFactor) *LR) * (((currentBatch - (maxBatches * fourthSector)) / ((maxBatches - (maxBatches * fourthSector)))));
        return thirdFactor * LR - x;
    }

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
        float slowLR = LR * (1.f - ((float)(currentBatch - tmp) / (float)(maxBatches - tmp + 1)));
        if (currentBatch > maxBatches / 2)
        {   
            float fastLR = LR * (1.f - (1.2f * ((float)(currentBatch - tmp) / (float)(maxBatches - tmp + 1))));
            if ( fastLR < 0.1f * LR)
            {
                if (slowLR < 0.1f* LR)
                {
                    return slowLR;
                }
                return 0.1f*LR;
            }
            return fastLR;
        }
        
        return slowLR;
    }
}