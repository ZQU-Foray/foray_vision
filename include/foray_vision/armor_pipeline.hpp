#ifndef FORAY_VISION_ARMOR_PIPELINE
#define FORAY_VISION_ARMOR_PIPELINE

#include "foray_vision/dual_detector.hpp"
#include "foray_vision/pnp_solver.hpp"
namespace foray_vision {
// 装甲板识别结果
struct ArmorObservation {
    ArmorDetection detection; // detect 层结果
    PnPResult pose;           // pnp 层结果
};
// 每一帧结果
struct VisionFrameResult {
    std::vector<ArmorObservation> armors;    // 仅 PnP 成功的装甲板，供下游使用
    std::vector<ArmorDetection> detections;  // 全部检测结果，保留原始顺序供连调
    std::vector<PnPResult> pnp_by_detection; // 与 detections 同下标；失败时 success=false
};
// 整合 detector 和 pnp

VisionFrameResult process_armor_solver(const std::vector<ArmorDetection>& det, const Solver& solver,
                                       const ArmorSize& small_armor_size);

VisionFrameResult process_armor_frame(DualDetector& detector, const Solver& solver,
                                      const ArmorSize& small_armor_size, const cv::Mat& bgr_img);

// 后续优化的(装甲板类型 全小!)(时间戳的采集)

} // namespace foray_vision

#endif
