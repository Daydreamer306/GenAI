# HW2：可审核的信息工程习题解答 MAS

Solver 生成草稿；Reviewer 独立检查数值、单位、证据和任务约束；Harness 驳回后反馈修订，
通过则停止，最多三轮。工具失败、证据缺失、模型错误、非法审核 JSON 或轮数耗尽都明确失败。
课程工具通过实际 MCP 调用；每次问答保存 trace、report、review、messages 和 metrics。

这是原 IE-Agent 小组项目的 HW2 扩展，当前采用自定义 harness，**不使用 AutoGen SDK**。
原成员：何铭源、郑一鸣、姚舜瑜、徐圣扬。新增贡献与限制见 [设计说明](docs/design.md)。

## 从新克隆开始

需要 Python 3.12 和 uv。默认使用 TF-IDF，无需 GPU、完整教材或 Qwen 权重。

```bash
git clone git@github.com:Daydreamer306/GenAI.git
cd GenAI/HW2
uv sync --locked --no-dev
uv run --no-dev python scripts/bootstrap_demo.py
uv run --no-dev ie-agent reproduce
```

`reproduce` 无需 key：调用真实工具，但 Solver/Reviewer 使用标为 simulated 的脚本模型，
演示“首次错误→驳回→修订→通过”，再比较六题的两种设计。模拟结果不是模型准确率实验。
建库脚本使用仓库中的自编小讲义，已有索引时不覆盖；它不是原项目完整教材库。

完整离线验证（36 项测试及七类成功/失败场景，预期失败的退出码也会检查）：

```bash
uv run --no-dev python scripts/verify_workflow.py
```

不使用 uv 时：新建 Python 3.12 虚拟环境，执行 `python -m pip install -e .`，
然后用 `python scripts/bootstrap_demo.py`、`ie-agent reproduce` 和
`python scripts/verify_workflow.py`。`requirements.txt` 为同一轻量依赖的版本范围；
精确复现依赖版本请使用 uv.lock。

## MiniMax 真实运行

把 `.env.example` 复制为本机 `.env`，填入你自己的 `MINIMAX_API_KEY`。不要提交 `.env`。
国内账户使用 `https://api.minimax.cn/v1`；默认模型为 MiniMax-M2.7。
MiniMax 配置缺失或失败不会自动调用旧 DeepSeek key。

```bash
cp .env.example .env
# 用编辑器填写 MINIMAX_API_KEY 后执行
uv run --no-dev ie-agent ask '计算概率 [0.5,0.5] 的信息熵' --planner rules
uv run --no-dev ie-agent ask '根据教材计算概率 [0.5,0.5] 的信息熵，并引用来源。' --planner rules
uv run --no-dev ie-agent benchmark
```

这里“教材”指 bootstrap 建立的自编演示资料，不是完整教材。
在线调用消耗 MiniMax 订阅额度或账户资源，运行前请确认自己的额度与支付设置。

一条命令重新测量两种设计并保存终端记录：

```bash
uv run --no-dev python scripts/verify_workflow.py --live
```

该命令只发送固定计算题及工具数值。明确同意发送少量检索资料时，可加 `--live-rag`。
模型具有随机性；可复现的是任务与测量程序，不保证再次得到相同答案或指标。

## Web 与 TUI

```bash
uv run --no-dev ie-agent web      # 默认仅绑定本机；端口及其他选项见 --help
uv run --no-dev ie-agent tui
```

`web/dist/` 已附构建页面，运行 Web 不需要 Node。
页面展示 trace 时间线、角色消息、每轮 verdict、停止原因、工具/引用和 token。
修改网页后，在 `web/` 中运行 `npm ci`、`npm run build` 和 `npm run lint`。
构建页面所用第三方组件的许可声明保留在 `web/THIRD_PARTY_LICENSES.txt`。

## 真实证据与测量结果

[artifacts/](artifacts/README.md) 保留精选真实 MiniMax 运行，不包含私有教材、密钥或模型权重。
六题固定 gold 下，单轮满足输出契约 5/6（83.3%），审核循环 6/6（100%）。
单轮调用 6 次，输入/输出 token 为 2933/2867；审核循环调用 12 次，token 为 7664/5382。
RC 题的 `1 ms` 数学等价于 `0.001 s`，但不符合约定 SI 单位，导致契约评测失分。
因此不能把差异解读为通用数学能力提升，也不保证审核通过的开放域文字完全正确。
另保留一个真实“驳回→修订→通过”的案例，详见归档说明。

新运行自动保存在 `runs/<run-id>/`，这个目录不入 Git，避免大量临时记录混入提交。
`artifacts/` 是明确精选、脱敏后纳入 Git 的历史证据。

## 可选：原 Qwen 检索路线

```bash
uv sync --locked --extra qwen
```

这会安装 torch、transformers、sentence-transformers 和 Chroma，下载量明显更大。
请自行提供有使用权限、带元数据的教材 Markdown，并在 `.env` 中明确设置
`IE_AGENT_KNOWLEDGE_DIR` 和 `IE_AGENT_RAG_BACKEND=qwen`，再用
`ie-agent knowledge ingest <资料目录> --backend qwen` 建库。
第一次使用可能下载 Qwen embedding 权重；已有本地模型与索引也可复用。
本次提交只验证轻量 TF-IDF 路线，**不声称 Qwen 后端已端到端复测**。

## 仓库结构与提交边界

```text
HW2/
  src/ie_agent/       Solver、Reviewer、harness、模型适配、RAG、MCP、工具及界面
  tests/             不依赖私有教材或真实 key 的自动测试
  examples/knowledge/ 自编小资料
  scripts/           安全建库与一键验证
  evaluation/        固定计算题 gold；旧教材检索题仅作历史参考
  docs/              作业原文、设计说明、提交清单
  artifacts/         精选真实运行及成本对比
  web/               前端源码、锁文件与构建页面
```

不上传 `.env`、虚拟环境、node_modules、完整教材、模型、数据库、临时 runs 或旧小组 PDF。
这些材料仍保留在原本地项目，不会因整理仓库而删除。
旧 `evaluation/retrieval_questions.yaml` 依赖完整教材，不能用自编小资料复现旧 Recall@K。
录屏和 2–4 页说明仍需通过课程渠道提交，见 [提交清单](docs/submission-checklist.md)。
