# pirl 项目对话变更记录

本文记录本轮项目审查和后续实现中完成的架构与工程变更。

## 执行层与循环守卫

- 模型侧只保留一个执行工具 `bash`(与 pi/dsh 一致)；`exec` 退为宿主内部通道，不再出现在 `ToolRegistry.specs()`。
- `bash` 标记为 `parallel_safe`，同一轮多个 bash/只读工具并发执行；写工具仍串行。
- `ExecutionPolicy` 不再用路径前缀正则判边界；命令中的绝对路径 `resolve()` 后与工作区比较，工作区位于 `/home` 下也不会误伤自己。
- 选中非 host 执行器(`rust`/`none`)时，`bash` 以 `bash -c <command>` 进入执行器，沙箱/禁用对模型真实生效。
- 循环守卫(重复调用、无新证据、无文件进展)不再被 `--max-steps` 关闭；命中先给一次转向提示，再命中才终止。
- `.pirl/` 从模型文件视野中移除(读写拒绝、列举/搜索跳过)。

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

## 上下文:journal / projection 重构

对齐 pi 的会话模型，把"在内存数组上压缩"换成"追加日志 + 派生视图":

- 新增 `pirl/session.py`:`SessionStore`(JSONL 追加写入) + `SessionManager`(entry 树、`parentId` 链、`projection()`/`build_context()`、`message_owners()`)。
- 压缩变成**追加一条 `compaction` entry**，携带 `summary` / `firstKeptEntryId` / `tokensBefore` / `details.readFiles|modifiedFiles` / `previousSummaryUsed`；原始 entry 全部保留，视图按边界重建。
- 重复压缩以 `previous-summary` 做增量更新，不再重蒸中段；边界指向旧 compaction 时自动收敛到它自己的边界，旧摘要不会叠加。
- 新增 `context_edit` entry(omit / replace)，作为不可恢复压缩的替代：只改模型视图，原始 entry 不变，历史可重新揭示。
- `ContextManager.prepare_view()` 返回 `ContextView`(messages + `CompactionRecord`)，Agent 把该决策落成 session entry，压缩因此可审计。
- **确定性**:`should_compact`(window - reserve)、`find_cut_point`(从新往旧累加、永不切在 tool result 上)、`estimate_context_tokens`(优先使用 provider 真实 usage，只估算尾部)全部是纯函数。
- **显式失败**:`SummarizationError` + `summarization_failure()` 拒绝 `finish_reason=length`、provider error、tool call、空摘要；`max_summary_failures` 熔断后走确定性有界尾部。
- **溢出恢复**:`stop_reason=length` 或 overflow 类错误触发一次 `compact_now` 重试，失败显式终止。
- `Completion` 增加 `stop_reason`；assistant entry 携带 `_usage`，仅在本地计数使用，发送前由 `strip_runtime_keys()` 剥离。
- `CheckpointStore` 增加 `identity` 与 `identity_mismatch()`：cwd/model/provider/tool_signature 不一致时不再静默 resume。
- CLI `run` 默认写 `.pirl/sessions/`，并新增 `--session <path>` 续跑同一 journal。

## 临时限制(待移除)

- **单次 run 最多 20 步模型调用**。在 `pirl/cli.py` 顶部:`TEMP_MAX_STEPS = 20`,`run --max-steps` 默认取该值，启动时向 stderr 打一行 `[limit] max 20 steps per run`，步数用尽时以 `max steps exceeded` 结束。
- 覆盖方式:`--max-steps N` 改上限；`--max-steps 0` 完全关闭。库层默认仍是无限(`agent.MAX_STEPS = None`)，不影响直接构造 `Agent` 的调用方与测试。
- 移除方式:删掉 `TEMP_MAX_STEPS`、把 `--max-steps` 的 `default` 改回 `None`,并删掉 `tests/test_cli.py`。

## 延迟:稳定前缀 / 少做重复劳动 / 流式

三项 P0 与三项 P1，目标是干掉"每轮重复劳动"而不是模型本身:

- **P0-1 进度指纹不再每轮全仓扫描**。`_workspace_fingerprint()` 一次 `rglob` 在 20k 文件仓库上 377ms，原先每轮调用两次。现在只在**本轮确实出现写类工具调用**时前后各算一次(read-only 轮零成本)；同时跳过 `node_modules/.venv/dist/target/build/vendor` 等目录。
- **P0-2 前缀字节稳定**。默认路径下**不再每轮重写更早的消息**：大输出外置的渲染只取决于该消息自身内容(含内容哈希)，因此旧消息永不漂移。重复观察折叠降级为 `ContextManager(dedupe=True)` 显式开关(默认关)——它必然重写一条更早的消息，从该位置起 prompt cache 全失效，省下的 token 不值一次重算。已带 `[COMPRESSED` 前缀的消息永不被二次处理。
- **P0-3 真实窗口**：`Agent(context_window=..., reserve_tokens=...)` 与 `keep_recent_tokens`；CLI 增加 `--context-window` / `--keep-recent-tokens`(env `PIRL_CONTEXT_WINDOW` / `PIRL_KEEP_RECENT_TOKENS`)，不再用 16000 硬编码。
- **P1-4 真 SSE 流式**：`ChatClient.stream()` 从"伪流式(complete 后一次性回吐)"改为解析 `text/event-stream`：文本增量 → `on_delta`，`reasoning_content/reasoning` 增量 → `on_reasoning`，tool_call 参数按 index 累加，`finish_reason` 与末包 `usage` 一并捕获。`Agent` 新增 `on_reasoning`；CLI 默认 `stream=True`，思考内容边生成边打印(stderr)；`--no-stream` 可关；结束后不再重复回放已流式输出的答案。
- **P1-5 摘要用独立模型**：`Agent(summarizer_client=...)`；CLI `--summarizer-model`(env `PIRL_SUMMARIZER_MODEL`)，摘要请求走更便宜/更快的模型，且天然不写 prompt cache。
- **P1-6 少绕一轮**：`BASE_SYSTEM` 明确要求"相互独立的读取/搜索请在同一个响应里一起发"，与已实现的并行工具执行对齐。

已知未覆盖:`is_split_turn`(单条 user span 超出 `keepRecentTokens` 时拆成两段摘要)暂未实现，`prepare_compaction` 目前只在消息边界切分。

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
