# 开发计划

## 任务列表

- [ ] 接装甲板检测/跟踪，产出带置信度的目标
- [x] 将 Detector 的接口、实现和测试迁入视觉包并完成本地构建
- [ ] 完成 Detector 输入/输出边界检查、模型验证及与 PnP 的调用链（纯数据调用链测试已通过，真实模型待验证）
- [ ] 实现多源态势融合（去重 / 坐标统一 / 时间对齐 / 置信度仲裁）
- [x] 实现双检测结果的原图像素框中心配对纯函数与歧义测试（尚未接入双模型推理）
- [x] 新增与旧 Detector 独立的双 ONNX 推理入口，离线实模加载和同帧推理通过
- [x] 双模型共享一次预处理并并行运行独立 Session；固定图片耗时基线与结果一致性测试已加入
- [x] 将双模型外轮廓四角到 SP 灯条端点的平面几何转换从模拟器移入 Vision，并给出尺寸显式输入与独立测试；仿真尺寸不可当作实车标定
- [x] 统一 Vision 输出类别为 -1/0/1/2；2 重装仅预留，未训练识别
- [ ] 复核双模型关键点标注顺序、置信度门限和颜色冲突；加入可调 HSV 仲裁，完成模拟器画面验收
- [ ] 生成 `WorldSnapshot`（对齐 `foray_interfaces` 的冻结 schema）
- [ ] 保证 `WorldSnapshot` 生成是纯函数、零 ROS 依赖

## 当前 Agent State

PLAN_READY

## Before Snapshot

commit hash:   （首次提交）
branch:        main
modified files: （首次初始化）
risk level:    L1

## 模糊点与待确认项

- 本仓接口尚未冻结，任务清单为**方向性**的，落地顺序以 `foray_docs/algorithm_structure.md` §9 演进阶段为准
- 依赖的自研仓尚未建齐，跨仓任务需等对方 `README.md` 明确接口后再启动

## Vector Backend Status

Backend: Markdown
Status:  ready
Environment: 仓库内 Markdown 文档（无外部向量后端）
Index: 本仓 `README.md` / `tree.md` / `decision.md`
Initialization: 2026-02-19
Commit: （首次提交）

## Acceptance Criteria

- [ ] 本仓能独立 `colcon build`（无代码时跳过）
- [x] 本地工作区中 `foray_vision` 可构建且 PnP/Detector/调用链单元测试通过
- [x] 双模型固定图分段计时与 CPU ORT 线程数对照完成；下一步需模拟器逐帧验证 Tracker 和端到端耗时
- [ ] CI 绿灯
- [ ] 每个新增模块带独立可执行测试
