#ifndef FORAY_VISION_PNP_SOLVER_HPP
#define FORAY_VISION_PNP_SOLVER_HPP

#include <cstdint>
#include <opencv2/core.hpp>

namespace foray_vision {
struct ArmorCorners {
    // 前面detcet规定 0=左上，1=右上，2=右下，3=左下
    cv::Point2f left_top_point;
    cv::Point2f right_top_point;
    cv::Point2f right_bottom_point;
    cv::Point2f left_bottom_point;
};

struct ArmorSize {
    float width;
    float height;
};

struct PnPResult {
    cv::Vec3d rvec_in_camera; // PnP 原始结果
    cv::Vec3d tvec_in_camera;
    cv::Matx33d R_armor2gimbal; // 装甲板系-云台系
    cv::Vec3d t_armor2gimbal;   // 装甲板中心在云台系的位置
    float reprojection_error{};
    bool success{false};
    // 平面靶标有镜像解：pnp_candidates=2 表示 IPPE 给了两个解；
    // pnp_ambiguous=true 表示没能用"看到正面"这一约束唯一确定朝向，下游应降级使用。
    int pnp_candidates{};
    bool pnp_ambiguous{true};
};
// 相机系到云台系
struct Camera2gimbal {
    cv::Matx33d R;
    cv::Vec3d t;
};

class Solver {
  public:
    explicit Solver(const cv::Mat& camera_matrix, const cv::Mat& distortion_coffs,
                    const Camera2gimbal& camera2gimbal);
    PnPResult solve(const ArmorCorners& corners, const ArmorSize& size) const;

  private:
    cv::Mat camera_matrix_;       // 相机内参
    cv::Mat distortion_coffs_;    // 畸变系数
    Camera2gimbal camera2gimbal_; // 相机to云台(r t)
};
/// @brief 下位机传输的实时yaw pitch
struct GimbalFeedback {
    double yaw;    // 左正 rad
    double pitch;  // 下正 rad
    int64_t stamp; // 时钟来源
    bool vaild;
};

// 云台到发射系 发射的放给下游处理
struct Gimbal2fire {
    cv::Matx33d R;
    cv::Vec3d t;
};

cv::Vec3d transform_point_c2g(const Camera2gimbal& Camera2gimbal, const cv::Vec3d& point_in_camera);

cv::Vec3d transform_point_g2f(const Gimbal2fire& Gimbal2fire, const cv::Vec3d& point_in_gimbal);

} // namespace foray_vision

#endif // FORAY_VISION_PNP_SOLVER_HPP
