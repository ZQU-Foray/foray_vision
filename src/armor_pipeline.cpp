#include "foray_vision/armor_pipeline.hpp"
#include "foray_vision/detector.hpp"
#include "foray_vision/pnp_solver.hpp"
// 调用层 调用 detect 和 pnp

namespace foray_vision {



    VisionFrameResult process_armor_solver(const std::vector<ArmorDetection>& detections ,const Solver& solver, const ArmorSize& small_armor_size)
    {   
            VisionFrameResult frame_result;
            frame_result.detections = detections;
            frame_result.pnp_by_detection.resize(detections.size());
            ArmorCorners corners;
            for (std::size_t i = 0; i < detections.size(); ++i)
            {
                const auto& detection = detections[i];
                if(detection.id < 0 || detection.id > 2 || detection.corners.size() != 4 )
                { 
                    continue;
                }
            // 注意顺序
            corners.left_top_point = detection.corners[0];
            corners.right_top_point = detection.corners[1];
            corners.right_bottom_point = detection.corners[2];
            corners.left_bottom_point = detection.corners[3];
            // pnp 层调用
            const auto pnp_result = solver.solve(corners, small_armor_size); // 默认都是小装甲板 
            frame_result.pnp_by_detection[i] = pnp_result;
            if(pnp_result.success == false)
            {
                continue;
            }
            ArmorObservation armorobservation;
            armorobservation.detection = detection;
            armorobservation.pose = pnp_result;
            frame_result.armors.push_back(armorobservation);
        }
        return frame_result;

    }
    // 前面两个函数的耦合
    VisionFrameResult process_armor_frame(Detector& detector, const Solver& solver, const ArmorSize& small_armor_size, const cv::Mat& bgr_img)
    {
        VisionFrameResult frame_result;   
        // detect 层调用
        const auto detections = detector.detect(bgr_img);
        // pnp 层调用
        frame_result = process_armor_solver(detections, solver, small_armor_size);

        return frame_result;

    }


}
