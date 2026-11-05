
#include "foray_vision/detector.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace foray_vision {

namespace {

    // 先使用常量 等待yaml文件再统一放进class 的private
// 模型输入边长 来源：训练配置 YOLO26-pose 640
constexpr int kInputSize = 640;
// letterbox 灰边填充值 必须与训练一致 来源：旧实现以前写的程序 armor_yolo_node.cpp
constexpr double kPadValue = 114.0;
// 候选置信度门槛 来源：旧实现（越低召回越高）
constexpr double kConfThr = 0.25;
// 同簇 IoU 门槛 来源：旧实现
constexpr double kIouThr = 0.5;
// 每簇最多保留候选数 来源：旧实现
constexpr int kKeepTop = 2;
// 类别数 B1..B5 BO BS / R1..R5 RO RS 来源：训练 data.yaml 参考自sp的训练
constexpr int kNumClasses = 14;
// 像素归一化除数
constexpr double kNormScale = 255.0;
// 模型输出行布局 共 26 行
constexpr int kBoxRows = 4;                        // cx cy w h
constexpr int kClassRow0 = kBoxRows;               // 类分行起点 = 4
constexpr int kKptRow0 = kClassRow0 + kNumClasses; // 角点行起点 = 18
constexpr int kNumKeypoints = 4;
constexpr int kNumRows = kKptRow0 + kNumKeypoints * 2; // = 26

// 类别名 顺序与训练一致 来源：旧实现 kClassNames
constexpr const char* kClassNames[kNumClasses] = {"B1", "B2", "B3", "B4", "B5", "BO", "BS",
                                                  "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
// {B1 B2 B3(重装) R1 R2 R3(重)}
std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) stream << ',';
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

// iou 框重合度 0-1
float iou(const Det& a, const Det& b) {
    const float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    const float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    const float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    const float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

    const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1); // 交集左上
    const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2); // 交集右下
    const float iw = std::max(0.f, ix2 - ix1);
    const float ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return (uni > 0.f) ? inter / uni : 0.f;
}

} // namespace

Detector::Detector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "foray_detector"),
     session_(env_, model_path.c_str(), Ort::SessionOptions{}) {
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
        throw std::runtime_error("Detector model must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_.GetInputNameAllocated(0, allocator);
    const auto output_name = session_.GetOutputNameAllocated(0, allocator);
    if (std::string(input_name.get()) != "images" || std::string(output_name.get()) != "output0") {
        throw std::runtime_error("Detector model nodes must be named images and output0");
    }

    const auto input_type_info = session_.GetInputTypeInfo(0);
    const auto output_type_info = session_.GetOutputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "Detector model input and output must use float tensors, got input type " +
            std::to_string(input_info.GetElementType()) + " and output type " +
            std::to_string(output_info.GetElementType()));
    }

    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    const std::vector<int64_t> expected_input{1, 3, kInputSize, kInputSize};
    if (input_shape != expected_input) {
        throw std::runtime_error("Detector model input shape must be [1,3,640,640], got " +
                                 shape_to_string(input_shape));
    }
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != kNumRows ||
        output_shape[2] <= 0) {
        throw std::runtime_error("Detector model output shape must be [1,26,N], got " +
                                 shape_to_string(output_shape));
    }
}

// 等比缩放 灰边处理 -640 x 640 画布 坐标统一左上角基准
cv::Mat Detector::letterbox(const cv::Mat& img, float& scale_out) const {
    const float scale = std::min((float)kInputSize / img.cols, (float)kInputSize / img.rows);
    scale_out = scale;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(cvRound(img.cols * scale), cvRound(img.rows * scale)));
    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3, cv::Scalar(kPadValue, kPadValue, kPadValue));
    // 缩放画面左上角 后面的坐标缩放也是
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));
    return canvas;
}
//  RGB - BGR 除以255归一化 hwc - CHW按照3种颜色分册
std::vector<float> Detector::to_tensor(const cv::Mat& canvas) const {
    if (canvas.type() != CV_8UC3) {
    throw std::invalid_argument("Detector canvas must be CV_8UC3 BGR");
}
    if (canvas.rows != 640 || canvas.cols != 640){
    throw std::invalid_argument("Detector canvas must be 640x640");
    }
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input(3 * kInputSize * kInputSize);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kInputSize; ++y) {
            for (int x = 0; x < kInputSize; ++x) {
                input[c * kInputSize * kInputSize + y * kInputSize +
                      x] = // CHW格式 channel heirgt width
                    rgb.at<cv::Vec3b>(y, x)[c] / kNormScale;
            }
        }
    }
    return input;
}
// 解码 按照输入数据集8400 还有data定义的26行的数据 将数据重处理并且检测conf
// 例子 : 取第 k 行第 i 列 = p[k * total + i]（行优先摊平：先跳 k 整行，再走 i 个）
std::vector<Det> Detector::decode(const float* p, int64_t total) const {
    std::vector<Det> dets;
    for (int64_t i = 0; i < total; i++) {
        float best = 0.f;
        int bestCls = -1;
        for (int k = 0; k < kNumClasses; ++k) {
            const float score =
                p[(kClassRow0 + k) * total + i]; // 倒顺序 这里是取每一个类别第i个先遍历类别再推进i
            if (score > best) {
                best = score;
                bestCls = k;
            }
        }
        if (best < kConfThr) {
            continue;
        }

        Det d;
        d.conf = best;
        d.cls = bestCls;
        d.cx = p[0 * total + i];
        d.cy = p[1 * total + i];
        d.w = p[2 * total + i];
        d.h = p[3 * total + i];
        // 列数代表模型输入的det数值
        // 行数是我们的类别 信心等等... 对应前面定义的起始行的定义
        for (int k = 0; k < kNumKeypoints; ++k) {
            // 这里的d.kpts[k] 是扁平数组 扁平数组和point2f的处理方式需要略微注意
            d.kpts[2 * k] = p[(kKptRow0 + 2 * k) * total + i];         // x
            d.kpts[2 * k + 1] = p[(kKptRow0 + 2 * k + 1) * total + i]; // y
        }
        dets.push_back(d);
    }
    return dets;
}

// nms 分簇
std::vector<int> Detector::nms_topk(const std::vector<Det>& dets) const {
    std::vector<std::vector<int>> clusters;
    for (size_t i = 0; i < dets.size(); ++i) {           // 多少个dets遍历外层
        int belong = -1;                                 // -1为没有簇
        for (int g = 0; g < (int)clusters.size(); ++g) { // 空的后面创组跟里面比较
            if (iou(dets[i], dets[clusters[g][0]]) >
                kIouThr) { // 跟第g簇的最高conf比较 前面要conf排序先
                belong = g;
                break;
            }
        }
        if (belong >= 0) {
            if ((int)clusters[belong].size() < kKeepTop) {
                clusters[belong].push_back((int)i);
            } // 相同组 满了2过就不要
        } else {
            clusters.push_back({(int)i}); // iou不同 前面conf过了 开新组
            
        }
    }

    std::vector<int> keep;
    for (const auto c : clusters) {
        for (auto idx : c) {
            keep.push_back(idx);
        }
    }
    return keep;
}
// 画框 - 用候选可视化
void draw(cv::Mat& canvas, const std::vector<Det>& dets, const std::vector<int>& kept) {
    for (int idx : kept) {
        const Det& d = dets[idx];
        const cv::Rect r(cvRound(d.cx - d.w / 2), cvRound(d.cy - d.h / 2), cvRound(d.w),
                         cvRound(d.h));
        cv::rectangle(canvas, r, cv::Scalar(0, 255, 0), 2); // 画框 绿色线
        cv::putText(canvas, kClassNames[d.cls], cv::Point(r.x, r.y - 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 255, 0), 2); // 画数据

        std::vector<cv::Point> poly; // 四点连线
        for (int k = 0; k < kNumKeypoints; ++k) {
            poly.push_back(cv::Point(cvRound(d.kpts[2*k]), cvRound(d.kpts[2*k + 1])));
        }
        cv::polylines(canvas, poly, true, cv::Scalar(0, 0, 255), 2); // 红色线
    }
}
// 计算方法 差积
float cross(cv::Point2f a, cv::Point2f b) {
    return a.x * b.y - a.y * b.x;
}
// 调用逻辑链路
std::vector<ArmorDetection> Detector::detect(const cv::Mat& bgr_img) {
    if(bgr_img.empty()){
        throw std::invalid_argument("Detector input image must not be empty");
    }
    if (bgr_img.type() != CV_8UC3) {
        throw std::invalid_argument("Detector bgr_img must be CV_8UC3 BGR");
    }
    // 预处理原图  640 画布（同时拿回缩放比）
    float scale = 0.0F;
    const cv::Mat canvas = letterbox(bgr_img, scale);
    // 画布 NCHW float 张量
    std::vector<float> input = to_tensor(canvas);
    // 推理
    std::array<int64_t, 4> in_shape{1, 3, kInputSize, kInputSize};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                 in_shape.data(), in_shape.size());
    const char* in_names[] = {"images"};
    const char* out_names[] = {"output0"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    //  取原始输出
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Detector inference must return one tensor");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = output_info.GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != kNumRows || shape[2] <= 0 ||
        output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementCount() % kNumRows != 0 ||
        output_info.GetElementCount() / kNumRows != static_cast<size_t>(shape[2])) {
        throw std::runtime_error("Detector inference output must be float [1,26,N], got " +
                                 shape_to_string(shape));
    }
    const float* p = outputs[0].GetTensorData<float>();

    //  解码 排序 去重
    std::vector<Det> dets = decode(p, shape[2]);          // shape[2] = 8400
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.conf > b.conf; });
    const std::vector<int> keep = nms_topk(dets);

    //  转成对外结构
    std::vector<ArmorDetection> result;
    for (int idx : keep) result.push_back(to_armor(dets[idx], scale));
    return result;

}

ArmorDetection Detector::to_armor(const Det& det, float scale) {
    ArmorDetection out{}; 

    // 左上角角点映射 
    out.corners.resize(kNumKeypoints);
    // 不循环 需要排列角点顺序
    // 模型 kpts：0=左上，1=左下，2=右下，3=右上
    // 输出 corners：0=左上，1=右上，2=右下，3=左下
    out.corners[0] = cv::Point2f(det.kpts[0] / scale, det.kpts[1] / scale);
    out.corners[1] = cv::Point2f(det.kpts[6] / scale, det.kpts[7] / scale);
    out.corners[2] = cv::Point2f(det.kpts[4] / scale, det.kpts[5] / scale);
    out.corners[3] = cv::Point2f(det.kpts[2] / scale, det.kpts[3] / scale);
    // conf
    out.confidence = det.conf;
    // 类型
    const int target_class = det.cls % 7;
    // 旧模型有具体车号；这里只保留当前目标类别，其余类别不进入目标跟踪。
    if (target_class >= 1 && target_class <= 5) {
        out.id = 1; // 步兵
    } else if (target_class == 6) {
        out.id = 0; // 哨兵
    }
    if (det.cls < 7) {
    out.color = ArmorColor::Blue;
} else {
    out.color = ArmorColor::Red;
}

    out.type = ArmorType::Small; // 只有等于0是大装甲板 默认是小装甲板 规则好像全是小装甲板
    if (target_class == 0) {out.type = ArmorType::Big;}

    return out;
}


} // namespace foray_vision
