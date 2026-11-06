# 目录结构说明

## 目录结构

```
.
├── README.md                本仓定位、接口、上下游
├── AGENTS.md                AI 协作规范
├── plan.md                  开发计划与进度
├── decision.md              关键工程决策日志
├── tree.md                  本文件
├── CODEOWNERS               Review 自动分派
├── .foray-layer             所属层与依赖边界（CI 校验）
├── .gitignore
├── CMakeLists.txt          PnP、双模型与调用链的构建、安装及单元测试
├── package.xml             ROS 2 包声明
├── include/foray_vision/  检测结果类型、PnP、双模型、调用链及图案框配对接口
├── src/                   PnP、双模型、调用链及图案框配对实现
├── test/                  对应的离线单元测试
└── .github/
    └── workflows/
        └── ci.yml           复制自组织 CI 模板
```

> 当前正式图像入口连接双模型与四角 PnP；旧单模型源码和测试已移除。双模型已完成真实 ONNX 离线推理，并共享预处理、并行执行两个独立 Session。HSV 仲裁、相机标定与实机验证仍待完成。

> `sp_lightbar_geometry` 是历史仿真适配工具；正式相机链路直接用物理外轮廓四角做 PnP，不转换为灯条端点。

> 双模型接口附带可选 CPU ORT 线程配置和单帧阶段计时；`test/test_dual_detector.cpp` 可用本地 ONNX/图片夹具做离线分段基准，`test/test_pnp.cpp` 可选重复解算计时。

## 记录约束

MUST NOT 记录以下内容：

- `build/`、`install/`、`log/`、`.git/` 等依赖与产物目录
- 临时文件、缓存文件、日志文件
