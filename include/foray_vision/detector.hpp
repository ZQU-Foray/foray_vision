#ifndef FORAY_VISION_DETECTOR_HPP
#define FORAY_VISION_DETECTOR_HPP

#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace foray_vision {

/// 装甲板规格 决定 PnP 用的真实尺寸
enum class ArmorType {
    Big,   ///< 大装甲板 英雄
    Small ///< 小装甲板 步兵 哨兵
};
// 装甲板灯光颜色
enum class ArmorColor {
  Blue,  //蓝
  Red    //红
};

/// 一块装甲板的检测结果 像素级 未解算
/// 坐标系 图像像素坐标 单位 像素
struct ArmorDetection {  
    std::vector<cv::Point2f> corners; ///< 4 个角点 顺序与 solver 的 3D 模型点一致
    int id = -1;                      ///< 目标类别：-1 无目标，0 哨兵，1 步兵，2 重装（预留）
    ArmorType type;                   ///< 装甲板规格
    double confidence;                ///< 置信度 0-1
    ArmorColor color;                 ///< 装甲板灯光颜色
};

// 模型输出的数据结构
struct Det {
    float conf = 0.f;                           // 置信度 = 14 类分里的最大值
    int cls = -1;                               // 类别号（0~13）
    float cx = 0.f, cy = 0.f, w = 0.f, h = 0.f; // 框（letterbox 空间的像素）
    float kpts[8] = {0.f};                      // 4 个角点 x,y（同空间）
};

/// 装甲板检测器 持有模型与阈值
class Detector {
  public:

    /// @param model_path ONNX模型文件路径
    explicit Detector(const std::string& model_path);

    /// @brief 由一帧图像检测装甲板
    /// @param bgr_img BGR 图像
    /// @return 本帧装甲板列表 空列表表示本帧无目标 像素级未解算
    std::vector<ArmorDetection> detect(const cv::Mat& bgr_img);

    /// @brief 等比缩放加灰边到模型输入画布
    /// @param img 原图
    /// @param scale_out 输出 缩放比例 用于把坐标映射回原图
    /// @return 输入画布
    cv::Mat letterbox(const cv::Mat& img, float& scale_out) const;

    /// @brief 画布转 NCHW float 张量 归一化 1/255
    std::vector<float> to_tensor(const cv::Mat& canvas) const;

    /// @brief 解析模型原始输出为候选
    /// @param data 输出数据 按 26 行 x total 列 排布
    /// @param total 候选数 例如 8400
    std::vector<Det> decode(const float* data, int64_t total) const;

    /// @brief 同簇去重 IoU 分簇 每簇保留前若干
    /// @param dets 已按置信度降序
    /// @return 保留候选的下标
    std::vector<int> nms_topk(const std::vector<Det>& dets) const;

    /// @brief 候选转装甲板结果 坐标映射回原图 类别映射 大小板判断
    /// 输出统一类别：-1 无目标，0 哨兵，1 步兵；2 重装暂无模型来源
    static ArmorDetection to_armor(const Det& det, float scale);

  private:

    Ort::Env env_;    ///< 模型环境
    Ort::Session session_; ///< 模型
};

} // namespace foray_vision

#endif // FORAY_VISION_DETECTOR_HPP
