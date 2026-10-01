# 精选运行证据

这些文件来自 2026-10-01 的实际运行，不是为了提交重新编造的结果。
原文件内 `artifact_dir` 指向当时的 `runs/`；此处按原 run_id 归档，内容不改写。

## 一次真实工具与多 Agent 运行

[20261001T120005-cee8fc3be8](runs/20261001T120005-cee8fc3be8/)：
MCP 实际计算信息熵，Solver 使用结果回答，Reviewer 审核通过。
包含 report.md、trace.jsonl、review.json、messages.json、metrics.json 等完整材料。

## 真实驳回和修订

[20261001T120314-91933fbcde](runs/20261001T120314-91933fbcde/)：
第一轮 RC 结果未满足 SI 单位输出契约，被硬审核门驳回，第二轮改为 0.001 s 后通过。
这是一个真实 MiniMax 修订案例，不是离线故障注入。

## 同题设计对比

- [六题汇总报告](live-comparison/report.md)
- [逐题 gold、结果、实际 token 与延迟](live-comparison/comparison.json)
- 对应的 12 次运行位于 `runs/<run_id>/`。

单轮输出契约满足率 5/6，审核循环 6/6；差异主要来自 RC 单位格式，
不能据此声称审核提高了通用数学能力。审核循环增加调用数、token 与延迟。

## 验证记录

- [终端文字记录](verification/terminal.log)
- [验证清单](verification/verification.json)

该记录包括 36 项测试、离线场景、真实计算题与六题真实对比。
标为 simulated 的运行使用模拟模型但真实工具，不能冒充真实 LLM 结果。
日志不是屏幕录像，仍需另外录屏。

提交整理后的新环境复测：[清单](submission-verification/verification.json)、
[终端记录](submission-verification/terminal.log)。重新建立虚拟环境，按更新的 uv.lock
导出默认依赖并从已校验的本机 wheel 缓存安装，未使用旧虚拟环境、原教材或真实 key。
36 项测试和七种成功/失败场景再次通过，真实 TF-IDF 测试使用自编小讲义。
依赖检查通过，提交版 Web 首页和 OpenAPI schema 均为 HTTP 200。

本次归档不包含完整教材、原教材检索片段、原索引、密钥或本机环境。
GitHub 可公开运行所需的自编小资料位于 `examples/knowledge/`。
