#include <cstdint>
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "foray_vision/pnp_solver.hpp"

using namespace foray_vision;

namespace {

// 只检查 R 是否为右手系的合法旋转；实际各轴朝向仍由坐标系约定决定。
bool is_right_handed_rotation(const cv::Matx33d& rotation)
{
    const cv::Matx33d should_be_identity = rotation.t() * rotation;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double expected = row == col ? 1.0 : 0.0;
            if (std::abs(should_be_identity(row, col) - expected) > 1e-9) {
                return false;
            }
        }
    }
    return std::abs(cv::determinant(rotation) - 1.0) < 1e-9;
}

} // namespace

TEST(PnPSolverTest, RecoversKnownPose)
{
float w = 0.3;
float h = 0.2; //装甲板系
cv::Vec3d rvec(0.2, 0.1, 0.3); // 旋转
cv::Vec3d tvec(1.0, 0.5, 2.0); // 平移
// 假设的内参
// clang-format off
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    800.0,   0.0, 640.0,
      0.0, 800.0, 512.0,
      0.0,   0.0,   1.0);
// clang-format on
// 假设成理想 没有畸变
cv::Mat distortion_coeffs = cv::Mat::zeros(1,5,CV_64F);

// 图像的角点 detect得出
std::vector<cv::Point2f> image_corners;
// 装甲板对应的点 3d装甲板系
// clang-format off
std::vector<cv::Point3f> object_points{
        {-w / 2, +h / 2, 0.0F},
        {+w / 2, +h / 2, 0.0F},
        {+w / 2, -h / 2, 0.0F},
        {-w / 2, -h / 2, 0.0F},
        };
// clang-format on
// 正向造数据
cv::projectPoints(object_points, rvec, tvec,camera_matrix, distortion_coeffs, image_corners);

const ArmorCorners corners{
    image_corners[0], // 左上
    image_corners[1], // 右上
    image_corners[2], // 右下
    image_corners[3], // 左下
};
// 方向验证解算
    // 测试用固定外参：绕 z 轴转 90°，再平移；不是实车标定参数。
    const cv::Matx33d R_camera2gimbal{0.0, -1.0, 0.0,
                                      1.0,  0.0, 0.0,
                                      0.0,  0.0, 1.0};
    const Camera2gimbal camera_to_gimbal{R_camera2gimbal, cv::Vec3d{0.1, -0.05, 0.02}};
    Solver solver(camera_matrix, distortion_coeffs, camera_to_gimbal);
    const ArmorSize armor_size{static_cast<float>(w), static_cast<float>(h)}; // double转float 
    const PnPResult result = solver.solve(corners, armor_size);
// 先验证是否解算成功
    ASSERT_TRUE(result.success);
    // 因为数据是造的用来验证模块所以 误差极小几乎为0
    EXPECT_NEAR(result.tvec_in_camera[0], tvec[0], 1e-3);
    EXPECT_NEAR(result.tvec_in_camera[1], tvec[1], 1e-3);
    EXPECT_NEAR(result.tvec_in_camera[2], tvec[2], 1e-3);
    EXPECT_NEAR(result.rvec_in_camera[0], rvec[0], 1e-3);
    EXPECT_NEAR(result.rvec_in_camera[1], rvec[1], 1e-3);
    EXPECT_NEAR(result.rvec_in_camera[2], rvec[2], 1e-3);
    EXPECT_NEAR(result.t_armor2gimbal[0], -0.4, 1e-3); // -0.5 + 0.1
    EXPECT_NEAR(result.t_armor2gimbal[1],  0.95, 1e-3); // 1.0 - 0.05
    EXPECT_NEAR(result.t_armor2gimbal[2],  2.02, 1e-3); // 2.0 + 0.02
    cv::Matx33d expected_R_armor2gimbal;
    cv::Rodrigues(rvec, expected_R_armor2gimbal);
    expected_R_armor2gimbal = R_camera2gimbal * expected_R_armor2gimbal;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            EXPECT_NEAR(result.R_armor2gimbal(row, col), expected_R_armor2gimbal(row, col), 1e-3);
        }
    }
    EXPECT_LT(result.reprojection_error, 1e-2);

    // 可选性能样本：同一组有效角点重复解算，避免把模型推理计入 PnP。
    if (const char* repeats_text = std::getenv("FORAY_TEST_PNP_BENCH_ITERS")) {
        const int repeats = std::atoi(repeats_text);
        ASSERT_GT(repeats, 0);
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            ASSERT_TRUE(solver.solve(corners, armor_size).success);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "PnP mean solve: " << ms / repeats << " ms (" << repeats
                  << " repeats)" << std::endl;
    }

};

TEST(PnPSolverTest, RejectsNaNCorner)
{
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 640.0,
        0.0, 800.0, 512.0,
        0.0, 0.0, 1.0);
    const cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(camera_matrix, distortion_coeffs,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});
    const ArmorSize armor_size{0.3F, 0.2F};

    ArmorCorners corners{
        {580.0F, 472.0F}, // 左上
        {700.0F, 472.0F}, // 右上
        {700.0F, 552.0F}, // 右下
        {580.0F, 552.0F}, // 左下
    };
    corners.left_top_point.x = std::numeric_limits<float>::quiet_NaN();

    const PnPResult result = solver.solve(corners, armor_size);
    EXPECT_FALSE(result.success);
}

TEST(PnPSolverTest, RejectsCoincidentCorners)
{
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 640.0,
        0.0, 800.0, 512.0,
        0.0, 0.0, 1.0);
    const cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(camera_matrix, distortion_coeffs,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});
    const ArmorCorners corners{
        {580.0F, 472.0F},
        {580.0F, 472.0F}, // 右上与左上重合
        {700.0F, 552.0F},
        {580.0F, 552.0F},
    };

    EXPECT_FALSE(solver.solve(corners, ArmorSize{0.3F, 0.2F}).success);
}

TEST(PnPSolverTest, RejectsCrossedCorners)
{
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 640.0,
        0.0, 800.0, 512.0,
        0.0, 0.0, 1.0);
    const cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(camera_matrix, distortion_coeffs,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});
    const ArmorCorners corners{
        {580.0F, 472.0F},
        {700.0F, 552.0F}, // 右上、右下交换后，连线自交 测试凸四边行
        {700.0F, 472.0F},
        {580.0F, 552.0F},
    };

    EXPECT_FALSE(solver.solve(corners, ArmorSize{0.3F, 0.2F}).success);
}

TEST(CoordinateTransformTest, CameraToGimbalAndFire) {

    const cv::Vec3d fire_origin_in_gimbal{-0.05, 0.0, 0.0};
    const cv::Vec3d target_in_camera{0.4, 0.0, 2.0}; // 目标在相机系
    const cv::Vec3d camera_origin_in_gimbal{0.1, 0.0, 0.0};
    const Camera2gimbal camera_to_gimbal{cv::Matx33d::eye(), camera_origin_in_gimbal};
    const cv::Vec3d target_in_gimbal = transform_point_c2g(camera_to_gimbal, target_in_camera);
    // 目标在发射系
    // t_fire_gimbal = -R_fire_gimbal * (发射系原点在云台系的位置)
    const Gimbal2fire gimbal_to_fire{cv::Matx33d::eye(), -fire_origin_in_gimbal};
    const cv::Vec3d target_in_fire = transform_point_g2f(gimbal_to_fire, target_in_gimbal);
    const cv::Matx33d rotation_gimbal_camera{
     0, 0, 1,
    -1, 0, 0,
     0,-1, 0
    };
    const cv::Vec3d target_ahead_in_camera{0.0, 0.0, 2.0};
    ASSERT_TRUE(is_right_handed_rotation(rotation_gimbal_camera));
    const Camera2gimbal rotated_camera_to_gimbal{rotation_gimbal_camera, camera_origin_in_gimbal};
    const cv::Vec3d target_in_gimbal_after_rotation =
        transform_point_c2g(rotated_camera_to_gimbal, target_ahead_in_camera);

    const double c = std::sqrt(3.0) / 2.0; // cos30
    const double s = 0.5;                  // sin30
    const cv::Matx33d rotation_30{
     c, 0, s,
     0, 1, 0,
    -s, 0, c
    };

    const cv::Vec3d fixed_target_in_gimbal{0.1, 0.0, 2.0};
    const cv::Vec3d seen_in_camera =
        rotation_30.t() * (fixed_target_in_gimbal - camera_origin_in_gimbal);

    const Camera2gimbal camera_to_gimbal_30{rotation_30, camera_origin_in_gimbal};
    const cv::Vec3d recovered_in_gimbal =
        transform_point_c2g(camera_to_gimbal_30, seen_in_camera);

    EXPECT_DOUBLE_EQ(target_in_gimbal_after_rotation[0], 2.1);
    EXPECT_DOUBLE_EQ(target_in_gimbal[0], 0.5);
    EXPECT_DOUBLE_EQ(target_in_fire[0], 0.55);
    EXPECT_DOUBLE_EQ(target_in_gimbal_after_rotation[2], 0.0);

    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(recovered_in_gimbal[axis], fixed_target_in_gimbal[axis], 1e-9);
    }

}

TEST(CoordinateTransformTest, AxisSignConvention)
{
    // 相机 x 右/y 下/z 前；云台 x 前/y 左/z 上（本测试的约定）。
    const cv::Matx33d rotation_gimbal_camera{
         0, 0, 1,
        -1, 0, 0,
         0,-1, 0
    };
    ASSERT_TRUE(is_right_handed_rotation(rotation_gimbal_camera));

    const Camera2gimbal camera_to_gimbal{rotation_gimbal_camera, cv::Vec3d{0.1, 0.0, 0.0}};
    const cv::Vec3d ahead_in_gimbal =
        transform_point_c2g(camera_to_gimbal, cv::Vec3d{0.0, 0.0, 2.0});
    EXPECT_NEAR(ahead_in_gimbal[0], 2.1, 1e-9); // 相机 +z 映射为云台 +x
    EXPECT_NEAR(ahead_in_gimbal[1], 0.0, 1e-9);
    EXPECT_NEAR(ahead_in_gimbal[2], 0.0, 1e-9);

    // 轴同向、发射系原点在云台系 x=-0.05：云台原点在发射系 x=+0.05。
    const Gimbal2fire gimbal_to_fire{cv::Matx33d::eye(), cv::Vec3d{0.05, 0.0, 0.0}};
    ASSERT_TRUE(is_right_handed_rotation(gimbal_to_fire.R));
    const cv::Vec3d ahead_in_fire = transform_point_g2f(gimbal_to_fire, ahead_in_gimbal);
    EXPECT_NEAR(ahead_in_fire[0], 2.15, 1e-9);

    // 镜像矩阵虽然会改变轴的正负，却不是合法的右手系旋转。
    const cv::Matx33d reflection{-1, 0, 0, 0, 1, 0, 0, 0, 1};
    EXPECT_FALSE(is_right_handed_rotation(reflection));
}

TEST(PnPSolverTest, PicksFrontFacingSolutionInsteadOfTheMirror)
{
    // 平面靶标的两个镜像解重投影几乎一致，但法向会翻 180°。
    // 我们只可能看到板正面，因此解必须满足 法向 · 板中心 < 0。
    const float w = 0.135F;
    const float h = 0.125F;
    const double yaw = 0.4;
    const cv::Matx33d rotate_z(std::cos(yaw), -std::sin(yaw), 0.0,
                               std::sin(yaw),  std::cos(yaw), 0.0,
                               0.0,            0.0,           1.0);
    // 绕 Y 轴 180°：板局部 +Z 从"背离相机"翻成"朝向相机"
    const cv::Matx33d rotate_y(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    const cv::Matx33d rotation = rotate_z * rotate_y;
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(rotation), rvec);
    const cv::Vec3d tvec(0.5, -0.2, 2.0);

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, 640.0,
        0.0, 800.0, 512.0,
        0.0, 0.0, 1.0);
    cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    const std::vector<cv::Point3f> object_points{
        {-w / 2, +h / 2, 0.0F}, {+w / 2, +h / 2, 0.0F},
        {+w / 2, -h / 2, 0.0F}, {-w / 2, -h / 2, 0.0F}};
    std::vector<cv::Point2f> image_corners;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs, image_corners);

    const Camera2gimbal identity{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}};
    const Solver solver(camera_matrix, distortion_coeffs, identity);
    const ArmorCorners corners{image_corners[0], image_corners[1],
                               image_corners[2], image_corners[3]};
    const PnPResult result = solver.solve(corners, ArmorSize{w, h});

    ASSERT_TRUE(result.success);
    const cv::Vec3d outward(result.R_armor2gimbal(0, 2), result.R_armor2gimbal(1, 2),
                            result.R_armor2gimbal(2, 2));
    // 1) 选中的解必须是"正面朝向相机"的那一支
    EXPECT_LT(outward.dot(result.tvec_in_camera), 0.0);
    // 2) 且与真值法向同向，而不是镜像
    const cv::Vec3d truth_outward(rotation(0, 2), rotation(1, 2), rotation(2, 2));
    EXPECT_GT(outward.dot(truth_outward), 0.99);
    // 3) 平移基本恢复
    EXPECT_NEAR(result.tvec_in_camera[0], tvec[0], 0.02);
    EXPECT_NEAR(result.tvec_in_camera[1], tvec[1], 0.02);
    EXPECT_NEAR(result.tvec_in_camera[2], tvec[2], 0.02);
    EXPECT_EQ(result.pnp_candidates, 2);
    EXPECT_FALSE(result.pnp_ambiguous);
}
