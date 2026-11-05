#include <gtest/gtest.h>

#include "foray_vision/sp_lightbar_geometry.hpp"

namespace {

TEST(SpLightbarGeometryTest, ConvertsOuterCornersToSpLightbarEndpoints)
{
    const std::array<cv::Point2f, 4> outer{{{0, 0}, {135, 0}, {135, 125}, {0, 125}}};
    const auto result = foray_vision::outer_to_sp_lightbars(
        outer, {0.135, 0.125, 0.135, 0.056});
    ASSERT_TRUE(result.has_value());
    const std::array<cv::Point2f, 4> expected{{{0, 34.5F}, {135, 34.5F},
                                                 {135, 90.5F}, {0, 90.5F}}};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR((*result)[i].x, expected[i].x, 0.01);
        EXPECT_NEAR((*result)[i].y, expected[i].y, 0.01);
    }
}

TEST(SpLightbarGeometryTest, RejectsImpossibleDimensions)
{
    const std::array<cv::Point2f, 4> outer{{{0, 0}, {135, 0}, {135, 125}, {0, 125}}};
    EXPECT_FALSE(foray_vision::outer_to_sp_lightbars(outer, {0.0, 0.125}));
    EXPECT_FALSE(foray_vision::outer_to_sp_lightbars(outer, {0.100, 0.0}));
}

} // namespace
