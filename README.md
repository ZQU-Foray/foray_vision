# foray_vision

> **L3** · owner 角色：视觉

目标感知 + 态势融合 + WorldSnapshot 生成

对外输出**唯一**的 `WorldSnapshot`，是决策层的共同输入。

---

## 快速开始

> 已搭建双模型装甲板检测、PnP 解算与逐板处理调用链，并完成本地构建及单元测试；真实相机标定和实机闭环仍待验证。

双模型入口对同帧图像只预处理一次，并行运行颜色/角点和图案两个独立 ONNX Runtime Session；仍保持同帧配对。当前仅使用现有 CPU 推理后端，性能需按部署设备复测。

正式图像入口 `process_armor_frame(DualDetector&, Solver&, ArmorSize, image)` 只接受双模型：颜色/外轮廓四角与图案唯一配对后，用外轮廓四角直接做 PnP。旧 `[1,26,N]` 单模型前端已移除；`sp_lightbar_geometry` 仅是历史仿真适配工具，真实链路不使用灯条端点。这里尚未接入实机相机采集、时间对齐或云台控制。

```bash
# 1. 拉取（仓齐备后改为从元仓 foray_ws 一键拉取）
git clone git@github.com:ZQU-Foray/foray_vision.git
cd foray_vision

# 2. 依赖安装
rosdep install -y -r --from-paths . --ignore-src --rosdistro humble

# 3. 构建
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. 运行
# （待装配根就绪后补充）
```

## 依赖

| 类型 | 依赖 |
|---|---|
| **自研仓** | X2、L1、L2、L3 |
| **第三方** | OpenCV · TensorRT · ROS 2 Humble |

## 接口

| 产出 | 内容 |
|---|---|
| `WorldSnapshot` | 区域级抽象状态（**非**原始传感器数据）· 纯函数 · 定长 |
| 目标感知 | 装甲板检测/跟踪结果 |
| 融合态势 | 多源去重 / 坐标统一 / 时间对齐 / 置信度仲裁 |

`WorldSnapshot` 字段分区（详见 `foray_interfaces`）：A `self` · B `allies[2]` · C `enemies[3]` ·
D `game` · E `context` · F `temporal`。

## `WorldSnapshot` 四条设计原则（违反任何一条都会破坏下游）



1. **定长**——不用变长数组（RL 需固定输入）；缺失用 `valid` 标志

2. **区域离散化优先**——用 `region_id` 代替原始坐标

   > 这是**关键耦合**：区域化状态 → 轻量对局仿真成为可能 → 红队 RL 可行

3. **带 `age` 与 `confidence`**——让决策器知道信息**多旧**、**多可信**

4. **纯函数、零 ROS 依赖**——可单元测试、可被仿真与实车共用



## 纪律



- `WorldSnapshot` 的 schema **必须与仿真输出完全一致**，否则离线闭环断裂

- 本仓是态势的**唯一权威来源**；其他仓不得自行拼凑世界状态

- 自瞄的目标感知细节在 `foray_auto_aim`，本仓只提供**装甲板级**的原始感知

## 上下游

| 方向 | 对象 |
|---|---|
| **上游**（本仓依赖谁） | `foray_interfaces` · `foray_platform` · `foray_localization` |
| **下游**（谁依赖本仓） | `foray_auto_aim` · `foray_decision` · `foray_robots` · `foray_ws` |

## 参考

- [算法结构](https://github.com/ZQU-Foray/foray_docs/blob/main/algorithm_structure.md)
- [仓库结构](https://github.com/ZQU-Foray/foray_docs/blob/main/repository_structure.md)
- [组织贡献指南](https://github.com/ZQU-Foray/.github/blob/main/CONTRIBUTING.md)
