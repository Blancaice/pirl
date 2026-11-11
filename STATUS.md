# pirl 状态记录

> 最后更新:compaction 完成(commit `a75899c`)。
> 参考:pi(`/home/blanca/pi-reference`)、dsh(`/home/blanca/deepseek-harness`)、
> ai-agent-book skills(`context-engineering` / `context-compression` /
> `loop-engineering` / `coding-agent-harness`)。

## 项目是什么

pirl 是一个本地 coding agent,核心是**可自生成的 skill 系统**:

- **A. skill 一等公民**(照抄 pi):发现 / 校验 / 渐进披露 / `/skill` 展开。
- **B. 自生成闭环**(自研):失败/重复 → 模型提案(带留出 tests)→ 验证门 → candidate → 人工激活 → 效用退役。
- 设计文档:`skill-system-selfgen-design.md`。

## 已确立的设计原则(来自参考,贯穿代码)

1. **静态前缀 + 追加**:system prompt 是有序 section(persona/rules/memory/skills),新内容**追加**、从不改写 persona(字节稳定,对 KV cache 友好)。
2. **确定性走代码**:能用代码强制的约束(路径隔离、验证、回退)不写成 prompt——"约束 > 指导"。
3. **模型能提出完成,但不能批准完成**:`finish(success=true)` 由框架跑验证命令 gate,失败驳回 + 结构化错误喂回。
4. **触发式/按需加载**:skill 正文、run 历史都是 name/摘要常驻、正文按需检索,不每次全量。
5. **结构化事件**:loop 只发 `StepStart`/`ToolCalled`/`FinishCalled`,渲染交给 CLI(`match` 分派)。
6. **压缩只压轨迹**:compaction 蒸馏旧 tool results,`messages[0]`(system 前缀)永不动;保留 decisions/failures/facts。

## 已完成(提交历史)

| commit | 内容 |
|---|---|
| `9eecd9e` | skill 系统(Part A 发现/校验/渐进披露/展开 + Part B 状态机/验证门/效用注册表) |
| `24cbab3` | agent 运行时:零依赖 LLM client(urllib)+ 5 工具 + turn loop + 真实 generate_skill |
| `48181c5` | finish 工具 + 效用统计自动接线(on_retrieve/on_adopt 自愈) |
| `a199658` | reasoning 显示 + 结构化进度事件 + 交互审核(reject/review)+ 系统提示 section 化 |
| `b2d1d41` | `ui.confirm/select`(TTY 感知) |
| `8a1c9d1` | 跨 run 记忆 + `.env` 加载 + 项目级 skill + LLM 重试 + `--verify` |
| `3f84d1a` | token 统计 + run 历史落盘(list_runs/read_run/search_history 按需检索) |
| `db38bda` | 循环内验证器(finish gate)+ git 快照回退(`restore`) |
| `a75899c` | compaction(阈值触发蒸馏旧 tool results) |

## 需要做的(按优先级)

### 高

- [ ] **流式输出(SSE)** —— 重排第 4 步:client 改 `stream: true`,loop 边收边渲染;顺带把 `reasoning` 升级成真 content block(pi 的 `ThinkingContent` / dsh 的 `ReasoningBlock`,含 delta 流 + 签名回放)。

### 中

- [ ] **专门 agent(preset/subagent)** —— "大方向二":preset = 模型 + 提示 + 工具子集 + skill 子集,派生子 agent(参考 pi 的 `examples/extensions/subagent/`)。
- [ ] **bash 沙箱** —— 工作区泄漏的硬解:bwrap(Linux)/ seatbelt(macOS),目前 bash 靠 prompt 约束,文件工具已用代码隔离。
- [ ] **run 内 compaction 五层补齐** —— 已有第 1 层(工具结果截断)和第 4 层(归档摘要);第 2 层(噪声删除)、第 5 层(全量 LLM 压缩 + 失败熔断)未做。

### 低

- [ ] **cost 估算** —— 需要按模型的价格表(`PIRL_COST_*`),目前只有 token 统计。
- [ ] **文档同步** —— README/设计文档随新功能持续更新(部分已做)。

## 已明确"暂不做 / 已覆盖"

- ~~前提 pre-flight prompt~~ → 已由**循环内验证器**(代码)覆盖。
- ~~跨 run 记忆(prompt 版)~~ → 已按"追加 section"重做。
- ~~reasoning 全量 content block 重构~~ → 留到流式一起做。

## CLI 命令一览

```
scan / prompt / expand          # Part A:发现 / 渐进披露 / 展开
init / ls / activate / reject / review / retire / verify   # Part B:生命周期
run [--cwd] [--max-steps] [--verify CMD]   # 端到端
restore <run_id> [--cwd]        # git 回退
```

## 配置(PIRL_* 环境变量,支持 .env)

```
PIRL_BASE_URL  PIRL_API_KEY  PIRL_MODEL
```
