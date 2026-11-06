#include "foray_vision/sp_lightbar_geometry.hpp"

#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace foray_vision {

std::optional<std::array<cv::Point2f, 4>>
outer_to_sp_lightbars(const std::array<cv::Point2f, 4>& outer_corners,
                      const SpLightbarGeometry& geometry) {
    if (!std::isfinite(geometry.outer_width_m) || geometry.outer_width_m <= 0.0 ||
        !std::isfinite(geometry.outer_height_m) || geometry.outer_height_m <= 0.0 ||
        !std::isfinite(geometry.lightbar_spacing_m) || geometry.lightbar_spacing_m <= 0.0 ||
        !std::isfinite(geometry.lightbar_length_m) || geometry.lightbar_length_m <= 0.0) {
        return std::nullopt;
    }
    for (const auto& corner : outer_corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y))
            return std::nullopt;
    }

    const std::vector<cv::Point2f> unit_square{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const std::vector<cv::Point2f> outer(outer_corners.begin(), outer_corners.end());
    const cv::Mat transform = cv::getPerspectiveTransform(unit_square, outer);
    if (transform.empty())
        return std::nullopt;

    const double half_u = geometry.lightbar_spacing_m / (2.0 * geometry.outer_width_m);
    const double half_v = geometry.lightbar_length_m / (2.0 * geometry.outer_height_m);
    const std::vector<cv::Point2f> lightbar_uv{
        {static_cast<float>(0.5 - half_u), static_cast<float>(0.5 - half_v)},
        {static_cast<float>(0.5 + half_u), static_cast<float>(0.5 - half_v)},
        {static_cast<float>(0.5 + half_u), static_cast<float>(0.5 + half_v)},
        {static_cast<float>(0.5 - half_u), static_cast<float>(0.5 + half_v)}};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(lightbar_uv, projected, transform);
    if (projected.size() != 4)
        return std::nullopt;
    std::array<cv::Point2f, 4> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(projected[i].x) || !std::isfinite(projected[i].y))
            return std::nullopt;
        result[i] = projected[i];
    }
    return result;
}

} // namespace foray_vision
