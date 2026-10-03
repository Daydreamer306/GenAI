# GenAI 课程作业

本仓库当前包含 **HW2：可审核的信息工程习题解答多智能体系统**。

- [运行说明](HW2/README.md)
- [系统设计与测量说明](HW2/docs/design.md)
- [作业原文](HW2/docs/assignment.md)
- [精选真实运行证据](HW2/artifacts/README.md)
- [提交检查表](HW2/docs/submission-checklist.md)

进入 `HW2/` 按 README 安装 Python 3.12 依赖即可运行。
默认采用轻量 TF-IDF，无需 GPU；在线模型默认 MiniMax，使用运行者自己的 key。
完整教材、模型权重、数据库、密钥及本地环境不上传，仅保存在本地。

系统包含 Solver/Reviewer 协作、受控审核循环、失败终态、证据保存和设计对比。
当前为自定义 harness，不使用 AutoGen SDK；是否接受这一实现需按课程要求确认。
