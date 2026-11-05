#include <gtest/gtest.h>

#include "foray_vision/pattern_match.hpp"

namespace foray_vision {
namespace {

TEST(PatternMatchTest, UsesPositionInsteadOfCountOrOrder) {
    const auto result = match_pattern_centers(
        {{0, 0, 20, 20}, {100, 0, 20, 20}, {200, 0, 20, 20}},
        {{105, 5, 4, 4}, {5, 5, 4, 4}});
    ASSERT_EQ(result.armor_to_pattern.size(), 3U);
    EXPECT_EQ(result.armor_to_pattern[0].state, MatchState::Matched);
    EXPECT_EQ(result.armor_to_pattern[0].index, 1U);
    EXPECT_EQ(result.armor_to_pattern[1].index, 0U);
    EXPECT_EQ(result.armor_to_pattern[2].state, MatchState::Unmatched);
}

TEST(PatternMatchTest, DoesNotForceOverlappingArmorCandidates) {
    const auto result = match_pattern_centers(
        {{0, 0, 20, 20}, {2, 0, 20, 20}}, {{8, 8, 4, 4}});
    EXPECT_EQ(result.pattern_to_armor[0].state, MatchState::Ambiguous);
    EXPECT_EQ(result.armor_to_pattern[0].state, MatchState::Unmatched);
    EXPECT_EQ(result.armor_to_pattern[1].state, MatchState::Unmatched);
}

TEST(PatternMatchTest, DoesNotForceTwoPatternsIntoOneArmor) {
    const auto result = match_pattern_centers(
        {{0, 0, 20, 20}}, {{2, 2, 4, 4}, {8, 8, 4, 4}});
    EXPECT_EQ(result.armor_to_pattern[0].state, MatchState::Ambiguous);
    EXPECT_EQ(result.pattern_to_armor[0].state, MatchState::Ambiguous);
    EXPECT_EQ(result.pattern_to_armor[1].state, MatchState::Ambiguous);
}

TEST(PatternMatchTest, KeepsMissingAndInvalidBoxesSeparate) {
    const auto result = match_pattern_centers(
        {{0, 0, 20, 20}, {30, 0, 0, 20}},
        {{50, 50, 4, 4}, {2, 2, -1, 4}});
    EXPECT_EQ(result.armor_to_pattern[0].state, MatchState::Unmatched);
    EXPECT_EQ(result.armor_to_pattern[1].state, MatchState::InvalidBox);
    EXPECT_EQ(result.pattern_to_armor[0].state, MatchState::Unmatched);
    EXPECT_EQ(result.pattern_to_armor[1].state, MatchState::InvalidBox);
}

} // namespace
} // namespace foray_vision
