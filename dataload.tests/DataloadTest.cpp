#include <gtest/gtest.h>

#include "DataLoad.hpp"
#include "MathLib.hpp"
using namespace dataload;

TEST(DataloadTest, DataLoadAllBasicTest)
{
    DataLoader dl("../dataload.tests/test_vector.csv", "../dataload.tests/test_labels.csv");
    std::vector<PicData> pd = dl.DataLoader::loadAllData(28,28);
    EXPECT_EQ(pd[2].getMat()(0,11) * 255, 105.0f);
    EXPECT_EQ(pd[2].getMat()(0,16) * 255, 132.0f);
    EXPECT_EQ(pd[1].getMat()(0,13) * 255, 5.0f);
    EXPECT_EQ(pd[1].getMat()(0,14) * 255, 6.0f);
    EXPECT_EQ(pd[1].getMat()(0,15) * 255, 7.0f);
}

TEST(DataloadTest, DataLoadOneLineBasicTest)
{
    DataLoader dl("../dataload.tests/test_vector.csv", "../dataload.tests/test_labels.csv");
    PicData pd = dl.loadPicture(28,28);
    EXPECT_EQ(pd.getMat()(0,8) * 255, 154.0f);
    EXPECT_EQ(pd.getMat()(0,22) * 255, 89.0f);
}
TEST(DataloadTest, DataLabelBasicTest)
{
    DataLoader dl("../dataload.tests/test_vector.csv", "../dataload.tests/test_labels.csv");
    std::vector<PicData> pd = dl.DataLoader::loadAllData(28,28);
    EXPECT_EQ(pd[0].getLabels()[2], 1.0f);
    EXPECT_EQ(pd[1].getLabels()[9], 1.0f);
    EXPECT_EQ(pd[2].getLabels()[6], 1.0f);
}