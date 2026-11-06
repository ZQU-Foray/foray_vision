#ifndef FORAY_VISION_ARMOR_DETECTION_HPP
#define FORAY_VISION_ARMOR_DETECTION_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace foray_vision {

/// 装甲板规格，决定 PnP 使用的真实尺寸。
enum class ArmorType {
    Big,
    Small
};

/// 装甲板灯光颜色。
enum class ArmorColor {
    Blue,
    Red
};

/// 双模型唯一配对后的像素级检测结果；四角顺序为左上、右上、右下、左下。
struct ArmorDetection {
    std::vector<cv::Point2f> corners;
    int id = -1; ///< 目标类别：-1 无目标，0 哨兵，1 步兵，2 重装（预留）
    ArmorType type = ArmorType::Small;
    double confidence = 0.0;
    ArmorColor color = ArmorColor::Blue;
};

} // namespace foray_vision

#endif // FORAY_VISION_ARMOR_DETECTION_HPP
