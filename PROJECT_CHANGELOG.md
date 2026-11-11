# pirl 项目对话变更记录

本文记录本轮项目审查和后续实现中完成的架构与工程变更。

## 初始架构审查

- 梳理了 `agent.py`、`tools.py`、`llm.py`、`compaction.py`、`lifecycle.py`、运行历史和 CLI 的职责边界。
- 识别出 coding agent 的主要风险：bash 并非真正 sandbox、完成判定可绕过验证、tool-call 压缩可能断链、状态与 runner 耦合、skill 成功率归因过粗。

## Agent loop 与验证

- `verify_command` 现在同时约束 `finish(success=true)` 和纯文本成功出口。
- Agent 实例复用时 token usage 按 run 重置。
- 新增可注入 `verifier`，并提供独立的 `CommandVerifier`。
- 新增取消检查：`cancel_event` 被设置后，run 以 `cancelled` 失败结果收尾。

## 上下文管理

- 修复 compaction 拆散 assistant `tool_calls` 与 `tool` result 的问题。
- 摘要估算和内容渲染现在包含 tool-call 字段。
- 新增可注入 `ContextManager`，runner 不再必须直接管理 compaction 策略。

## 工具系统

- `ToolResult` 增加 `ok` 和稳定的 JSON message envelope。
- 新增 workspace 内 `apply_patch` 工具，优先支持定向修改，减少整文件覆盖。
- `search` 优先使用 `rg`，无 `rg` 时回退纯 Python 实现。
- bash 增加 `ExecutionPolicy`，默认拦截常见破坏性、网络和明显工作区外命令。
- bash 超时会终止整个进程组，避免子进程残留。
- 明确记录：ExecutionPolicy 不是 OS 级 sandbox，宿主可注入真正隔离执行器。

## Skill 归因与 hooks

- `retrievals` 与 explicit application 分离。
- 新增 `adopt_skill` 工具，只有 skill 已召回且模型提供非空 reason 时才记录 application。
- application 支持 success、failure、unknown 统计，读取 skill 本身不再自动计为采用。
- 新增 `SkillUsageHook`，skill 持久化统计从 runner 主流程中抽离。
- 保留 `SkillStore.on_adopt` 作为兼容别名，内部转发到 explicit application。

## 并行、流式与可恢复执行

- 同一模型响应中的普通 tool calls 支持 `ThreadPoolExecutor` 并行执行，可通过 `parallel_tools` 和 `max_tool_workers` 控制。
- finish call 仍保持终止语义，不与普通工具并行提交。
- `ChatClient.stream` 提供 streaming-compatible 接口；当前兼容客户端回退为单个最终 delta，provider 可覆盖为 SSE/增量实现。
- Agent 支持 `on_text` 回调。
- 新增 `CheckpointStore`，使用临时文件 + `os.replace` 原子保存/读取 `.pirl/checkpoints/<run_id>.json`。
- Agent 支持 checkpoint callback，在每个 step 后持久化消息和 trace。

## 文件恢复与持久化安全

- git snapshot 保存 tracked dirty diff，restore 后恢复运行前的 tracked 修改。
- snapshot 存在 pre-existing untracked 文件时，restore 会拒绝执行，避免静默丢失内容。
- run history 写盘前脱敏常见 API key、token、password、secret 和 Bearer 凭据。

## 文档与验证

- README 测试数量同步更新。
- 回归测试从初始状态扩展到当前 `186 passed`。
- 未修改用户已有的未跟踪文件 `STATUS.md`。

## 尚需宿主配合的边界

- 真正的进程级文件、网络和权限隔离仍需由宿主提供容器、sandbox 或等价 executor。
- 当前 `ChatClient.stream` 是接口兼容层，不代表所有 provider 已启用真实 SSE 流式传输。

## 当前验证

- 最后一次全量验证：`189 passed`。

## 安全默认值补充

- CLI `pirl run` 现在默认关闭 bash；需要 shell 时必须显式传 `--allow-bash`。
- 原因是 `cwd=workspace` 只能改变当前目录，不能阻止 `../../`、symlink 或 shell 子命令访问 workspace 外部路径。

## Rust execution kernel

- 新增 `rust-exec/` crate：`WorkspaceGuard`、argv-only `Supervisor`、timeout/process-group cleanup、output caps、CPU/地址空间 rlimits、offline network namespace（创建失败则 fail-closed）、patch path validation、atomic checkpoint journal。
- Rust 通过 JSONL 支持 `exec`、`patch`、`journal` 操作。
- Python 新增 `RustExecutor` / `executor_from_env` adapter；设置 `PIRL_EXECUTOR` 后可由宿主切换到 Rust executor。
- 当前环境没有安装 `rustc/cargo`，因此 crate 尚未在本机编译；Python 测试仍已通过。

## Context / tools / memory priority pass

- verifier 在配置 Rust executor 时优先走 argv-only executor，拒绝 shell syntax。
- memory 和 tool result 注入增加 `trust="untrusted"` / source 标记，提醒模型其内容是数据而非系统指令。
- 并行 tool calls 只允许显式标记 `parallel_safe` 的只读工具；写入、bash、patch 默认串行。
- Agent 支持从 versioned checkpoint state 的 messages/trace 恢复继续运行。
- 新增对应回归测试；最后一次 Python 全量验证为 `191 passed`。
- Rust crate 已补充 5 个安全单元测试：workspace 越界、patch 越界、argv 执行、输出截断、超时；用户环境此前的 `cargo test` 之所以显示 0 tests，是因为之前尚未加入这些测试。
