#include <cstdint>
#include <gtest/gtest.h>

#include "foray_vision/detector.hpp"

using namespace foray_vision;

namespace {
constexpr double kTolerance = 1e-6;
}

TEST(DetectorToArmor, 蓝方四号角点与字段映射正确) {
    Det det{};
    det.cls = 3; // B4
    det.conf = 0.9F;

    // 模型角点顺序：左上、左下、右下、右上。
    det.kpts[0] = 20.0F;
    det.kpts[1] = 40.0F;
    det.kpts[2] = 20.0F;
    det.kpts[3] = 100.0F;
    det.kpts[4] = 140.0F;
    det.kpts[5] = 100.0F;
    det.kpts[6] = 140.0F;
    det.kpts[7] = 40.0F;

    const auto armor = Detector::to_armor(det, 2.0F);

    EXPECT_EQ(armor.id, 1);
    EXPECT_EQ(armor.color, ArmorColor::Blue);
    EXPECT_EQ(armor.type, ArmorType::Small);
    EXPECT_NEAR(armor.confidence, 0.9, kTolerance);

    ASSERT_EQ(armor.corners.size(), 4U);
    EXPECT_NEAR(armor.corners[0].x, 10.0, kTolerance);
    EXPECT_NEAR(armor.corners[0].y, 20.0, kTolerance);
    EXPECT_NEAR(armor.corners[1].x, 70.0, kTolerance);
    EXPECT_NEAR(armor.corners[1].y, 20.0, kTolerance);
    EXPECT_NEAR(armor.corners[2].x, 70.0, kTolerance);
    EXPECT_NEAR(armor.corners[2].y, 50.0, kTolerance);
    EXPECT_NEAR(armor.corners[3].x, 10.0, kTolerance);
    EXPECT_NEAR(armor.corners[3].y, 50.0, kTolerance);
}

TEST(DetectorToArmor, 红蓝同兵种编号相同颜色不同) {
    Det blue{};
    blue.cls = 3; // B4

    Det red{};
    red.cls = 10; // R4

    const auto blue_armor = Detector::to_armor(blue, 1.0F);
    const auto red_armor = Detector::to_armor(red, 1.0F);

    EXPECT_EQ(blue_armor.id, 1);
    EXPECT_EQ(red_armor.id, 1);
    EXPECT_EQ(blue_armor.color, ArmorColor::Blue);
    EXPECT_EQ(red_armor.color, ArmorColor::Red);
}

TEST(DetectorToArmor, 旧模型不支持的目标为无目标) {
    Det hero{};
    hero.cls = 0;
    EXPECT_EQ(Detector::to_armor(hero, 1.0F).id, -1);
    Det sentry{};
    sentry.cls = 6;
    EXPECT_EQ(Detector::to_armor(sentry, 1.0F).id, 0);
    Det infantry_five{};
    infantry_five.cls = 5;
    EXPECT_EQ(Detector::to_armor(infantry_five, 1.0F).id, 1);
}
