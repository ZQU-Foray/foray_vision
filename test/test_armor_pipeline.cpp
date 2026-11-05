#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include <vector>

#include "foray_vision/armor_pipeline.hpp"

// 造数据 + 验证
namespace foray_vision {
namespace {
// 模拟装甲板长宽
constexpr ArmorSize kArmorSize{0.3F, 0.2F};
//  模拟相机内参
cv::Mat camera_matrix() {
    return (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 640.0,
        0.0, 800.0, 512.0,
        0.0, 0.0, 1.0);
}
// 造数据 
ArmorDetection make_detection(int id, const cv::Vec3d& translation,
                              const cv::Mat& intrinsic, const cv::Mat& distortion) {
    const float half_width = kArmorSize.width / 2.0F;
    const float half_height = kArmorSize.height / 2.0F;
    const std::vector<cv::Point3f> object_points{
        {-half_width, +half_height, 0.0F},
        {+half_width, +half_height, 0.0F},
        {+half_width, -half_height, 0.0F},
        {-half_width, -half_height, 0.0F},
    };

    ArmorDetection detection{};
    detection.id = id;
    detection.type = ArmorType::Small;
    detection.color = ArmorColor::Red;
    detection.confidence = 0.9;
    // 重投影点 检测目标角点 旋转矩阵 平移向量 相机内参 畸变系数 图片点
    cv::projectPoints(object_points, cv::Vec3d{0.0, 0.0, 0.0}, translation,
                      intrinsic, distortion, detection.corners);
    return detection;
}

TEST(ArmorPipelineTest, EmptyDetectionsReturnEmptyFrame) {
    const cv::Mat intrinsic = camera_matrix();
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(intrinsic, distortion,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});

    const VisionFrameResult result = process_armor_solver({}, solver, kArmorSize);
    EXPECT_TRUE(result.armors.empty());
    EXPECT_TRUE(result.detections.empty());
    EXPECT_TRUE(result.pnp_by_detection.empty());
}

TEST(ArmorPipelineTest, KeepsValidArmorsAroundInvalidDetection) {
    const cv::Mat intrinsic = camera_matrix();
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(intrinsic, distortion,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});

    const ArmorDetection first = make_detection(0, cv::Vec3d{0.0, 0.0, 2.0},
                                                 intrinsic, distortion);
    ArmorDetection invalid = make_detection(-1, cv::Vec3d{0.2, 0.0, 2.0},
                                             intrinsic, distortion);
    invalid.corners.pop_back();
    const ArmorDetection last = make_detection(1, cv::Vec3d{0.4, 0.0, 2.5},
                                                intrinsic, distortion);

    const VisionFrameResult result = process_armor_solver({first, invalid, last},
                                                           solver, kArmorSize);
    ASSERT_EQ(result.armors.size(), 2U);
    ASSERT_EQ(result.detections.size(), 3U);
    ASSERT_EQ(result.pnp_by_detection.size(), 3U);
    EXPECT_TRUE(result.pnp_by_detection[0].success);
    EXPECT_FALSE(result.pnp_by_detection[1].success);
    EXPECT_TRUE(result.pnp_by_detection[2].success);
    EXPECT_EQ(result.armors[0].detection.id, 0);
    EXPECT_EQ(result.armors[1].detection.id, 1);
    EXPECT_TRUE(result.armors[0].pose.success);
    EXPECT_TRUE(result.armors[1].pose.success);
    EXPECT_NEAR(result.armors[0].pose.tvec_in_camera[2], 2.0, 1e-3);
    EXPECT_NEAR(result.armors[1].pose.tvec_in_camera[2], 2.5, 1e-3);
}

} // namespace
} // namespace foray_vision
