#ifndef FORAY_VISION_PATTERN_MATCH_HPP
#define FORAY_VISION_PATTERN_MATCH_HPP

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace foray_vision {

// 两路模型的框必须来自同一帧，并且都已映射回原始图像像素坐标。
enum class MatchState { Unmatched, Matched, Ambiguous, InvalidBox };

struct PatternMatch {
    MatchState state = MatchState::Unmatched;
    std::size_t index = 0; // 仅 state == Matched 时有效
};

struct PatternMatchResult {
    // 每个装甲板框对应一个结果；歧义时不强行选置信度最高的图案。
    std::vector<PatternMatch> armor_to_pattern;
    // 每个图案框对应一个结果；可定位漏检、重叠框和坏框。
    std::vector<PatternMatch> pattern_to_armor;
};

PatternMatchResult match_pattern_centers(const std::vector<cv::Rect2f>& armor_boxes,
                                         const std::vector<cv::Rect2f>& pattern_boxes);

} // namespace foray_vision

#endif
