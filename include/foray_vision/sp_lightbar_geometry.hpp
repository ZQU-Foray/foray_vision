#ifndef FORAY_VISION_SP_LIGHTBAR_GEOMETRY_HPP
#define FORAY_VISION_SP_LIGHTBAR_GEOMETRY_HPP

#include <array>
#include <optional>

#include <opencv2/core.hpp>

namespace foray_vision {

/// 双模型标注的装甲板外轮廓转换为 SP PnP 使用的灯条端点。
/// 两组角点顺序均为左上、右上、右下、左下；外轮廓尺寸须由标注契约或实测给出。
/// 135 mm 灯条间距与 56 mm 灯条长度来自 SP 小装甲板求解器，非本赛季实车标定值。
struct SpLightbarGeometry {
    double outer_width_m;
    double outer_height_m;
    double lightbar_spacing_m = 0.135;
    double lightbar_length_m = 0.056;
};

std::optional<std::array<cv::Point2f, 4>> outer_to_sp_lightbars(
    const std::array<cv::Point2f, 4>& outer_corners,
    const SpLightbarGeometry& geometry);

} // namespace foray_vision

#endif
