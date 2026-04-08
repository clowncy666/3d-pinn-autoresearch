# 3D PINN 自动研究协议 (AutoResearch Protocol - Time-Marching 版)

你的目标是通过不断修改代码，尽可能降低这个 3D PINN 模型的 `FINAL_AUTORESEARCH_SCORE`（这是一个结合了全局 MAE 和局部最大误差的综合评分）。

## 实验规则 (无限循环)：
请严格、无限次地重复以下步骤，不要停下来问我，直接执行：

1. **分析与假设**：思考一下修改什么可以降低综合分数。注意，当前采用的是**时间域分解 (Time-Marching)** 架构，你的优化方向可以包括但不限于：
   - Phase 1 (0-6s) 和 Phase 2 (6-7s) 的独立学习率和衰减策略。
   - Phase 2 独有的 Soft IC Loss (初始条件软约束) 权重调整。
   - 两个阶段各自的 Adam / L-BFGS 训练步数分配。
   - 各个子域（Substrate/Bump/Chip）在不同阶段的采样点数量调整等。
2. **修改代码**：直接编辑 `代码.py` 落实你的假设。如果涉及到推断逻辑的改变，必须同步修改 `error.py`。
3. **运行训练**：在终端执行 `D:\cy\芯片基板pinn\venv\Scripts\python.exe 代码.py`，等待运行结束。
4. **运行评估**：执行 `D:\cy\芯片基板pinn\venv\Scripts\python.exe error.py`，并精准读取终端最后一行打印的 `FINAL_AUTORESEARCH_SCORE: [数值]`。
5. **裁判与结算**：
   ⚠️ **铁律**：执行所有 git 命令时，**绝对禁止使用 `cd` 命令**。必须使用 `git -C` 参数指定工作目录，分两步执行：
   - 如果当前的综合分数 **低于** 历史最低记录：
     这说明你的修改有效！依次执行（必须同时追踪训练和评估代码）：
     第一步：`git -C "D:\cy\芯片基板pinn\三维\三维1.0.1" add 代码.py error.py`
     第二步：`git -C "D:\cy\芯片基板pinn\三维\三维1.0.1" commit -m "Improvement: [简述你改了什么]，分数降至 [填入数值]"`
   - 如果综合分数 **变大** 了，或者运行报错了：
     说明方向错了！执行：
     `git -C "D:\cy\芯片基板pinn\三维\三维1.0.1" reset --hard`
6. **循环**：回到步骤 1，换一个新思路继续！不要停止！