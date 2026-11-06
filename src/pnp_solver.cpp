#include "foray_vision/pnp_solver.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace foray_vision {
Solver::Solver(const cv::Mat& camera_matrix, const cv::Mat& distortion_coffs,
               const Camera2gimbal& camera2gimbal)
    : camera_matrix_(camera_matrix.clone()), distortion_coffs_(distortion_coffs.clone()),
      camera2gimbal_(camera2gimbal) {
}
PnPResult Solver::solve(const ArmorCorners& corners, const ArmorSize& size) const {
    PnPResult result{};
    // 检查输入
    const bool valid_width = std::isfinite(size.width) && size.width > 0.0F;
    const bool valid_height = std::isfinite(size.height) && size.height > 0.0F;
    if (!valid_height || !valid_width) {
        return result; // 默认false
    }
    // W H 是大小装甲板由调用层再改size里面的参数注入 大小装甲板放在匿名namespace里面
    float w = size.width;
    float h = size.height;
    // 3d坐标是以板中心为原点 z始终设置成z = 0
    // 角点顺序记得对应
    std::vector<cv::Point3f> object_points{
        {-w / 2, +h / 2, 0.0F},
        {+w / 2, +h / 2, 0.0F},
        {+w / 2, -h / 2, 0.0F},
        {-w / 2, -h / 2, 0.0F},
    };
    // Armorcorner是上游的数据detect传入 调用层处理数据
    std::vector<cv::Point2f> image_points{
        corners.left_top_point,
        corners.right_top_point,
        corners.right_bottom_point,
        corners.left_bottom_point,
    };
    // 检验图片角点 是否合格(无限大)
    for (int i = 0; i < 4; ++i) {
        bool vaild_img_point_x = std::isfinite(image_points[i].x);
        bool vaild_img_point_y = std::isfinite(image_points[i].y);
        if (!vaild_img_point_x || !vaild_img_point_y) {
            return result;
        }
    }

    // 四个点互不重合
    for (std::size_t i = 0; i < image_points.size(); ++i) {
        for (std::size_t j = i + 1; j < image_points.size(); ++j) {
            if (image_points[i] == image_points[j]) {
                return result;
            }
        }
    }
    // 且按约定顺序围成 凸四边形
    if (cv::contourArea(image_points) <= 0.0 || !cv::isContourConvex(image_points)) {
        return result;
    }

    cv::Mat rvec;
    cv::Mat tvec;

    // 平面靶标在 IPPE 下有两个镜像解（板法向分别朝向/背向相机）。此时两者重投影
    // 几乎一样，单看重投影误差选不出来，但它会让下游"板朝向"整体翻 180°。
    // 这里显式取出全部解，用"我们只看到板的正面"这一物理约束挑：
    // 板法向 n=R*(0,0,1) 与板中心 t 的夹角必须 > 90°，即 n·t < 0。
    // 若没有解满足（例如严重遮挡/角点乱序），退回两者中重投影误差较小者，保持旧行为。
    std::vector<cv::Mat> candidate_rvecs;
    std::vector<cv::Mat> candidate_tvecs;
    const int candidate_count =
        cv::solvePnPGeneric(object_points, image_points, camera_matrix_, distortion_coffs_,
                            candidate_rvecs, candidate_tvecs, false, cv::SOLVEPNP_IPPE);

    if (candidate_count > 0) {
        int front_facing = -1;    // 满足"看到正面"且重投影最小的解
        int best_error_index = 0; // 纯重投影最小的解（兜底）
        double best_front_error = std::numeric_limits<double>::max();
        double best_error = std::numeric_limits<double>::max();
        for (int i = 0; i < candidate_count; ++i) {
            const auto index = static_cast<std::size_t>(i);
            cv::Matx33d rotation;
            cv::Rodrigues(candidate_rvecs[index], rotation);
            const cv::Vec3d translation(candidate_tvecs[index]);
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points, candidate_rvecs[index], candidate_tvecs[index],
                              camera_matrix_, distortion_coffs_, projected);
            double error = 0.0;
            for (std::size_t k = 0; k < image_points.size(); ++k)
                error += cv::norm(projected[k] - image_points[k]);
            error /= static_cast<double>(image_points.size());

            if (error < best_error) {
                best_error = error;
                best_error_index = i;
            }
            const cv::Vec3d outward(rotation(0, 2), rotation(1, 2), rotation(2, 2));
            if (outward.dot(translation) < 0.0 && error < best_front_error) {
                best_front_error = error;
                front_facing = i;
            }
        }
        result.pnp_candidates = candidate_count;
        result.pnp_ambiguous = candidate_count > 1 && front_facing < 0;
        const int chosen = front_facing >= 0 ? front_facing : best_error_index;
        rvec = candidate_rvecs[static_cast<std::size_t>(chosen)];
        tvec = candidate_tvecs[static_cast<std::size_t>(chosen)];
    } else {
        const bool solved = cv::solvePnP(object_points, image_points, camera_matrix_,
                                         distortion_coffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
        if (!solved) {
            return result;
        } // 默认返回false
        result.pnp_candidates = 0;
        result.pnp_ambiguous = true; // 连多解接口都失败，朝向不可信
    }

    result.rvec_in_camera = cv::Vec3d{
        rvec.at<double>(0, 0),
        rvec.at<double>(1, 0),
        rvec.at<double>(2, 0),
    };
    result.tvec_in_camera = cv::Vec3d{
        tvec.at<double>(0, 0),
        tvec.at<double>(1, 0),
        tvec.at<double>(2, 0),
    };

    std::vector<cv::Point2f> projected_points; // 根据r 和 t再用真实装甲板尺寸预测出的角点
    cv::projectPoints(object_points,           // 装甲板系 我称为实际
                      rvec, tvec, camera_matrix_, distortion_coffs_, projected_points);
    // 相机系到云台系
    cv::Matx33d R_armor2camera;
    cv::Rodrigues(rvec, R_armor2camera); // 转化到cv的
    cv::Matx33d R_armor2gimbal = camera2gimbal_.R * R_armor2camera;
    cv::Vec3d t_armor2gimbal = camera2gimbal_.R * result.tvec_in_camera + camera2gimbal_.t;
    result.R_armor2gimbal = R_armor2gimbal;
    result.t_armor2gimbal = t_armor2gimbal;

    float error_sum = 0.0F; // 重投影差总和 只有4个映射点的差
    for (std::size_t i = 0; i < image_points.size(); ++i) {
        error_sum += cv::norm(projected_points[i] - image_points[i]);
    }
    result.reprojection_error = error_sum / static_cast<float>(image_points.size());
    result.success = true;

    return result;
}

cv::Vec3d transform_point_c2g(const Camera2gimbal& Camera2gimbal,
                              const cv::Vec3d& point_in_camera) {
    return Camera2gimbal.R * point_in_camera + Camera2gimbal.t;
}

cv::Vec3d transform_point_g2f(const Gimbal2fire& Gimbal2fire, const cv::Vec3d& point_in_gimbal) {
    return Gimbal2fire.R * point_in_gimbal + Gimbal2fire.t;
}

} // namespace foray_vision
