#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "foray_vision/dual_detector.hpp"
#include "foray_vision/armor_pipeline.hpp"

namespace foray_vision {
TEST(DualDetectorTest, RunsTwoRealModelsWhenFixtureIsProvided) {
    const char* pose = std::getenv("FORAY_TEST_POSE_ONNX");
    const char* pattern = std::getenv("FORAY_TEST_PATTERN_ONNX");
    const char* image = std::getenv("FORAY_TEST_IMAGE");
    if (!pose || !pattern || !image)
        GTEST_SKIP() << "Set three local fixture paths";
    const cv::Mat bgr = cv::imread(image);
    ASSERT_FALSE(bgr.empty());
    const char* threads_text = std::getenv("FORAY_TEST_INTRA_THREADS");
    const int intra_op_threads = threads_text ? std::stoi(threads_text) : 0;
    DualDetector detector(pose, pattern, intra_op_threads);
    const auto frame = detector.detect(bgr);
    EXPECT_GT(frame.pose_candidates, 0U);
    EXPECT_GT(frame.pattern_candidates, 0U);
    EXPECT_EQ(frame.matches.armor_to_pattern.size(), frame.pose_candidates);
    EXPECT_EQ(frame.matches.pattern_to_armor.size(), frame.pattern_candidates);
    const auto repeated = detector.detect(bgr);
    EXPECT_EQ(repeated.pose_candidates, frame.pose_candidates);
    EXPECT_EQ(repeated.pattern_candidates, frame.pattern_candidates);
    ASSERT_EQ(repeated.armors.size(), frame.armors.size());
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        800.0, 0.0, bgr.cols / 2.0, 0.0, 800.0, bgr.rows / 2.0, 0.0, 0.0, 1.0);
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    const Solver solver(camera_matrix, distortion,
                        Camera2gimbal{cv::Matx33d::eye(), cv::Vec3d{0.0, 0.0, 0.0}});
    const auto pipeline = process_armor_frame(detector, solver, ArmorSize{0.135F, 0.055F}, bgr);
    EXPECT_EQ(pipeline.detections.size(), frame.armors.size());
    EXPECT_EQ(pipeline.pnp_by_detection.size(), pipeline.detections.size());
    for (std::size_t i = 0; i < frame.armors.size(); ++i) {
        EXPECT_EQ(repeated.armors[i].id, frame.armors[i].id);
        EXPECT_EQ(repeated.armors[i].color, frame.armors[i].color);
        ASSERT_EQ(repeated.armors[i].corners.size(), frame.armors[i].corners.size());
        for (std::size_t k = 0; k < frame.armors[i].corners.size(); ++k)
            EXPECT_LE(cv::norm(repeated.armors[i].corners[k] - frame.armors[i].corners[k]), 1e-4);
    }
    if (const char* repeats_text = std::getenv("FORAY_TEST_BENCH_ITERS")) {
        const int repeats = std::atoi(repeats_text);
        ASSERT_GT(repeats, 0);
        const auto started = std::chrono::steady_clock::now();
        DualDetector::Timing sum;
        for (int i = 0; i < repeats; ++i) {
            const auto timing = detector.detect(bgr).timing;
            sum.preprocess_ms += timing.preprocess_ms;
            sum.pose_inference_ms += timing.pose_inference_ms;
            sum.pose_postprocess_ms += timing.pose_postprocess_ms;
            sum.pattern_inference_ms += timing.pattern_inference_ms;
            sum.pattern_postprocess_ms += timing.pattern_postprocess_ms;
            sum.match_ms += timing.match_ms;
            sum.total_ms += timing.total_ms;
        }
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        std::cout << "DualDetector mean detect: " << elapsed / repeats << " ms (" << repeats
                  << " repeats); stages: preprocess=" << sum.preprocess_ms / repeats
                  << " pose_run=" << sum.pose_inference_ms / repeats
                  << " pose_post=" << sum.pose_postprocess_ms / repeats
                  << " pattern_run=" << sum.pattern_inference_ms / repeats
                  << " pattern_post=" << sum.pattern_postprocess_ms / repeats
                  << " match=" << sum.match_ms / repeats
                  << " measured_total=" << sum.total_ms / repeats << std::endl;
    }
}
} // namespace foray_vision
