#include "foray_vision/dual_detector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <stdexcept>
#include <vector>

namespace foray_vision {
namespace {

constexpr int kSide = 640;
constexpr float kConfidence = 0.25F;
constexpr float kNmsIou = 0.5F;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

Ort::SessionOptions session_options(int intra_op_threads) {
    if (intra_op_threads < 0)
        throw std::invalid_argument("intra_op_threads must be nonnegative");
    Ort::SessionOptions options;
    if (intra_op_threads > 0)
        options.SetIntraOpNumThreads(intra_op_threads);
    return options;
}

struct Candidate {
    cv::Rect2f box;
    float score = 0.0F;
    int cls = -1;
    std::array<cv::Point2f, 4> points{};
    std::array<float, 4> visibility{};
};

float overlap(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float area = (a & b).area();
    const float total = a.area() + b.area() - area;
    return total > 0.0F ? area / total : 0.0F;
}

std::vector<Candidate> suppress(std::vector<Candidate> candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
        const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const Candidate& prior) {
            return overlap(candidate.box, prior.box) > kNmsIou;
        });
        if (!duplicate)
            kept.push_back(candidate);
    }
    return kept;
}

std::vector<float> prepare(const cv::Mat& bgr, float& scale) {
    scale = std::min(static_cast<float>(kSide) / bgr.cols, static_cast<float>(kSide) / bgr.rows);
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(cvRound(bgr.cols * scale), cvRound(bgr.rows * scale)));
    cv::Mat canvas(kSide, kSide, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> tensor(3 * kSide * kSide);
    constexpr std::size_t plane = static_cast<std::size_t>(kSide) * kSide;
    float* red = tensor.data();
    float* green = red + plane;
    float* blue = green + plane;
    for (int y = 0; y < kSide; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        const std::size_t offset = static_cast<std::size_t>(y) * kSide;
        for (int x = 0; x < kSide; ++x) {
            red[offset + x] = row[x][0] / 255.0F;
            green[offset + x] = row[x][1] / 255.0F;
            blue[offset + x] = row[x][2] / 255.0F;
        }
    }
    return tensor;
}

} // namespace

class DualDetector::Model {
  public:
    struct Output {
        std::vector<Candidate> candidates;
        double inference_ms = 0.0;
        double postprocess_ms = 0.0;
    };

    Model(const std::string& path, int rows, int intra_op_threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "foray_dual_detector"),
          session_(env_, path.c_str(), session_options(intra_op_threads)), rows_(rows) {
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
            throw std::runtime_error("DualDetector expects one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
        // TypeInfo 必须活到读取形状之后；不能从临时 TypeInfo 留下悬空视图。
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto output_type = session_.GetOutputTypeInfo(0);
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        const auto output = output_type.GetTensorTypeAndShapeInfo();
        const auto shape = output.GetShape();
        if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input.GetShape() != std::vector<int64_t>({1, 3, kSide, kSide}) ||
            output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 3 ||
            shape[0] != 1 || shape[1] != rows_ || shape[2] <= 0)
            throw std::runtime_error("DualDetector model tensor contract mismatch");
    }

    Output run(std::vector<float>& input, float scale) {
        const std::array<int64_t, 4> input_shape{1, 3, kSide, kSide};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                      input_shape.data(), input_shape.size());
        const char* input_name = input_name_.c_str();
        const char* output_name = output_name_.c_str();
        const auto inference_started = Clock::now();
        auto outputs =
            session_.Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1, &output_name, 1);
        const double inference_ms = elapsed_ms(inference_started);
        const auto postprocess_started = Clock::now();
        if (outputs.size() != 1 || !outputs[0].IsTensor())
            throw std::runtime_error("DualDetector did not return one tensor");
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 3 ||
            shape[0] != 1 || shape[1] != rows_ || shape[2] <= 0 ||
            info.GetElementCount() != static_cast<std::size_t>(rows_ * shape[2]))
            throw std::runtime_error("DualDetector runtime output shape mismatch");
        const float* data = outputs[0].GetTensorData<float>();
        const auto count = shape[2];
        const auto at = [&](int row, int64_t col) { return data[row * count + col]; };
        std::vector<Candidate> candidates;
        for (int64_t i = 0; i < count; ++i) {
            const float a = at(4, i), b = at(5, i);
            const int cls = a >= b ? 0 : 1;
            const float score = std::max(a, b);
            if (!std::isfinite(score) || score < kConfidence)
                continue;
            const float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
                !std::isfinite(h) || w <= 0 || h <= 0)
                continue;
            Candidate candidate;
            candidate.box = {(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale};
            candidate.score = score;
            candidate.cls = cls;
            if (rows_ == 18) {
                bool valid_points = true;
                for (int k = 0; k < 4; ++k) {
                    const float x = at(6 + 3 * k, i), y = at(7 + 3 * k, i);
                    candidate.visibility[k] = at(8 + 3 * k, i);
                    if (!std::isfinite(x) || !std::isfinite(y) ||
                        !std::isfinite(candidate.visibility[k]))
                        valid_points = false;
                    candidate.points[k] = {x / scale, y / scale};
                }
                if (!valid_points)
                    continue;
            }
            candidates.push_back(candidate);
        }
        return {suppress(std::move(candidates)), inference_ms, elapsed_ms(postprocess_started)};
    }

  private:
    Ort::Env env_;
    Ort::Session session_;
    int rows_;
    std::string input_name_, output_name_;
};

DualDetector::DualDetector(const std::string& color_pose_model, const std::string& pattern_model,
                           int intra_op_threads)
    : pose_(std::make_unique<Model>(color_pose_model, 18, intra_op_threads)),
      pattern_(std::make_unique<Model>(pattern_model, 6, intra_op_threads)) {
}

DualDetector::~DualDetector() = default;

DualDetector::Frame DualDetector::detect(const cv::Mat& bgr_image) {
    const auto started = Clock::now();
    if (bgr_image.empty() || bgr_image.type() != CV_8UC3)
        throw std::invalid_argument("DualDetector requires a nonempty CV_8UC3 BGR image");
    // 两个模型使用同一张、同一预处理契约的图；只生成一次输入张量。
    float scale = 0.0F;
    auto input = prepare(bgr_image, scale);
    DualDetector::Frame frame;
    frame.timing.preprocess_ms = elapsed_ms(started);
    // 两个独立 Session 可并行推理。input 在两个任务完成前保持有效；
    // future.get() 还会把模型异常带回调用线程，避免静默丢帧。
    auto pattern_task = std::async(std::launch::async, [&] { return pattern_->run(input, scale); });
    const auto pose_output = pose_->run(input, scale);
    const auto pattern_output = pattern_task.get();
    const auto& poses = pose_output.candidates;
    const auto& patterns = pattern_output.candidates;
    frame.timing.pose_inference_ms = pose_output.inference_ms;
    frame.timing.pose_postprocess_ms = pose_output.postprocess_ms;
    frame.timing.pattern_inference_ms = pattern_output.inference_ms;
    frame.timing.pattern_postprocess_ms = pattern_output.postprocess_ms;
    const auto match_started = Clock::now();
    std::vector<cv::Rect2f> pose_boxes, pattern_boxes;
    for (const auto& pose : poses)
        pose_boxes.push_back(pose.box);
    for (const auto& pattern : patterns)
        pattern_boxes.push_back(pattern.box);
    frame.pose_candidates = poses.size();
    frame.pattern_candidates = patterns.size();
    frame.matches = match_pattern_centers(pose_boxes, pattern_boxes);
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const auto match = frame.matches.armor_to_pattern[i];
        if (match.state != MatchState::Matched)
            continue;
        const auto& pose = poses[i];
        // 可见性低的角点不能偷偷送入 PnP；侧板/出画面后续需单独策略。
        if (std::any_of(pose.visibility.begin(), pose.visibility.end(),
                        [](float value) { return value < 0.5F; }))
            continue;
        ArmorDetection armor{};
        armor.corners.assign(pose.points.begin(), pose.points.end());
        armor.color = pose.cls == 0 ? ArmorColor::Blue : ArmorColor::Red;
        armor.id = patterns[match.index].cls; // 图案模型：0 哨兵、1 步兵；重装=2 暂无模型输出
        armor.type = ArmorType::Small;
        armor.confidence = std::min(pose.score, patterns[match.index].score);
        frame.armors.push_back(std::move(armor));
    }
    frame.timing.match_ms = elapsed_ms(match_started);
    frame.timing.total_ms = elapsed_ms(started);
    return frame;
}

} // namespace foray_vision
