#include "foray_vision/pattern_match.hpp"

#include <cmath>

namespace foray_vision {
namespace {

bool valid_box(const cv::Rect2f& box) {
    return std::isfinite(box.x) && std::isfinite(box.y) &&
           std::isfinite(box.width) && std::isfinite(box.height) &&
           box.width > 0.0F && box.height > 0.0F &&
           std::isfinite(box.x + box.width) &&
           std::isfinite(box.y + box.height);
}

} // namespace

PatternMatchResult match_pattern_centers(
    const std::vector<cv::Rect2f>& armor_boxes,
    const std::vector<cv::Rect2f>& pattern_boxes) {
    PatternMatchResult result;
    result.armor_to_pattern.resize(armor_boxes.size());
    result.pattern_to_armor.resize(pattern_boxes.size());
    std::vector<std::size_t> pattern_counts(armor_boxes.size(), 0);

    for (std::size_t armor = 0; armor < armor_boxes.size(); ++armor) {
        if (!valid_box(armor_boxes[armor])) {
            result.armor_to_pattern[armor].state = MatchState::InvalidBox;
        }
    }

    for (std::size_t pattern = 0; pattern < pattern_boxes.size(); ++pattern) {
        const auto& box = pattern_boxes[pattern];
        if (!valid_box(box)) {
            result.pattern_to_armor[pattern].state = MatchState::InvalidBox;
            continue;
        }

        const cv::Point2f center{box.x + box.width / 2.0F,
                                 box.y + box.height / 2.0F};
        std::size_t containing_count = 0;
        std::size_t containing_armor = 0;
        for (std::size_t armor = 0; armor < armor_boxes.size(); ++armor) {
            if (valid_box(armor_boxes[armor]) && armor_boxes[armor].contains(center)) {
                ++containing_count;
                containing_armor = armor;
            }
        }
        if (containing_count > 1) {
            result.pattern_to_armor[pattern].state = MatchState::Ambiguous;
        } else if (containing_count == 1) {
            result.pattern_to_armor[pattern] = {MatchState::Matched, containing_armor};
            ++pattern_counts[containing_armor];
            result.armor_to_pattern[containing_armor] = {MatchState::Matched, pattern};
        }
    }

    for (std::size_t armor = 0; armor < armor_boxes.size(); ++armor) {
        if (pattern_counts[armor] > 1) {
            result.armor_to_pattern[armor] = {MatchState::Ambiguous, 0};
        }
    }
    for (std::size_t pattern = 0; pattern < pattern_boxes.size(); ++pattern) {
        const auto match = result.pattern_to_armor[pattern];
        if (match.state == MatchState::Matched && pattern_counts[match.index] > 1) {
            result.pattern_to_armor[pattern] = {MatchState::Ambiguous, 0};
        }
    }
    return result;
}

} // namespace foray_vision
