# pirl — Skill 系统(A 一等公民 + B 自生成闭环)

按 `skill-system-selfgen-design.md` 实现,分两部分:

- **A. skill 一等公民** —— 照抄 pi(`packages/coding-agent/src/core/skills.ts`)的
  语言无关移植。
- **B. 自生成闭环** —— pi 没有,按设计文档自研(创新点)。

## 目录结构

```
pirl/
├── __init__.py      # 公共 API 再导出
├── frontmatter.py   # A1/A4: SKILL.md frontmatter 解析(与 pi 行为一致)
├── ignore.py        # A3: 依赖自由的 gitignore 风格匹配器
├── skills.py        # A2/A3/A4: Skill 数据模型 + 发现 + 校验 + load_skills
├── prompt.py        # A5: 渐进式披露 <available_skills>
├── commands.py      # A5: /skill:name args 展开
├── lifecycle.py     # B1/B2: 生命周期状态机 + 侧车注册表(skill-registry.json)
├── proposal.py      # B3/B4/B5: 触发源 + 提案结构 + 验证门(留出样例)
├── loop.py          # B6: 自生成闭环 + 效用统计 + 定期退役
├── llm.py           # 零依赖 OpenAI 兼容 chat client(urllib)
├── generate.py      # 真实 generate_skill:模型输出 → Proposal(JSON 契约 + 解析 + 校验)
├── tools.py         # 工作区文件工具 + 单一执行工具 bash + ToolRegistry
├── agent.py         # turn loop:section 化系统提示 + 工具执行 + trace + 结构化进度事件 + 自生成
├── session.py       # append-only JSONL journal:entry 树 / projection / context_edit / compaction
├── context.py       # 派生模型视图:去重 → 哈希外置 → 高水位摘要(含确定性回退)
├── checkpoint.py    # 单文件快照 + identity 校验(主 journal 已改为 session.py)
├── plugins/compaction.py # 确定性切点 / token 计数 / 结构化摘要 / 失败判定
├── run_memory.py    # 跨 run 记忆(追加 section,不碰 persona)
├── env.py           # 最小 .env 加载(不覆盖已 export 的变量,值按字面量)
├── ui.py            # ui.confirm / ui.select(TTY 感知,非交互回退)
└── __main__.py      # CLI(scan / prompt / expand / init / ls / activate / retire / reject / review / verify / run)
tests/               # pytest 测试(192 个用例)
```

## 运行时增强

- **系统提示 = 有序 section**:`persona` / `rules` / `memory` / `skills`,新内容**追加**、
  从不改写 persona(字节级稳定,对 KV cache 友好)。
- **结构化进度事件**:`StepStart` / `ToolCalled` / `FinishCalled`,loop 只发事件,
  CLI 用 `match` 渲染——渲染层与循环解耦。
- **思考显示**:`reasoning_content` / `reasoning` 收集到 `RunResult.reasoning`,
  `run` 打到 stderr(需推理模型,如 DeepSeek `deepseek-reasoner`)。
- **跨 run 记忆**:`<cwd>/.pirl/last-run.md` 记录上一次 outcome+答复,下次作为
  `memory` section 注入,治"重复劳动"。
- **`.env` 加载**:`run` 启动时 `load_dotenv()`,已 export 的变量优先、值不展开。
- **项目级 skill**:`run` 除 `~/.pirl/skills/active` 外也扫 `<cwd>/.pirl/skills`(按名去重)。
- **LLM 重试**:429/5xx/超时指数退避 + 抖动,尊重 `Retry-After`。
- **确定性验证**:`run --verify "python -m pytest"` 在任务后用退出码判定成败。
- **token 统计**:跨步骤累加 `usage`(prompt/completion/total),结束打到 stderr。
- **Append-only session journal**:`.pirl/sessions/<ts>_<id>.jsonl` 是唯一事实来源——消息、compaction 边界、context edit、模型切换、usage 全部追加写入，永不重写。每轮模型输入由 `SessionManager.projection()` 从当前分支派生，因此压缩可逆、可审计、可回放。
- **三层上下文压缩**:L1 大输出与旧写入正文按哈希外置到 `.pirl/objects/`(可重新取回,渲染只取决于该消息自身内容 → 永不漂移)；L2 重复观察折叠为 `dedupe=True` 显式开关(**默认关**——它会重写更早的消息,从该位置起 prompt cache 全部失效,省下的 token 不值一次重算)；L3 仅在 `window - reserveTokens` 高水位后用 LLM 蒸馏旧轨迹。**切点、token 计数、边界全部确定**，只有摘要文本来自模型；摘要若 `finish_reason=length`/provider 报错/试图调工具/为空，一律拒绝落盘(`SummarizationError`)并退回确定性有界尾部，`max_summary_failures` 后熔断。system 前缀永不进入摘要。
- **前缀字节稳定(为 KV cache 服务)**:已发送过的消息在后续请求中渲染为相同字节,除非一次 compaction 整段替换。这是 prompt cache 能命中的前提；任何"每轮重算旧消息"的优化都会把 prefill 成本乘回去。
- **流式 + 实时思考**:`ChatClient.stream()` 走真 SSE(`stream=true`),文本增量走 `on_text`、思考增量走 `on_reasoning`;CLI 默认开启,思考内容随生成实时显示,不再等到结束后回放。
- **显式失败路径**:provider 报 context overflow 或 `stop_reason=length` 时允许**恰好一次** compact-and-retry；失败则保留 journal 原样、不追加 compaction，并以 `context overflow recovery failed: ...` 终止，不做隐式截断。
- **交互审核**:`run` 生成候选后询问 `[a]ctivate/[r]eject/[s]kip`;另有 `review` / `reject` 命令。
- **单一执行工具**:模型只看到 `bash`(和 pi/dsh 一样),`exec(argv)` 仅作为宿主内部通道保留(验证器、插件),不再发给模型——两套执行原语并存时模型会在两者之间来回试探。
- **并发工具调用**:同一轮里的多个只读工具或 `bash` 会并发执行(`max_tool_workers`,默认 4);`write_file`/`apply_patch` 仍串行,混入写调用的批次整体串行。
- **循环守卫常开**:重复调用、无新证据、无文件进展都会先给一次转向提示("停止调查,要么 finish,要么说清还缺哪个检查"),再犯即终止;`--max-steps` 只是外层上限,不再把守卫关掉。
- **`.pirl/` 对模型不可见**:文件工具拒绝读写 `.pirl/`(session journal、run 历史、object store),`list_files`/`search` 也跳过它——否则模型会把自己的历史当成任务线索,越查越像有内幕。
- **路径边界按真实路径判定**:命令里的绝对路径先 `resolve()` 再看是否在工作区内,而不是拿 `/home`、`/etc` 之类前缀正则硬匹配(工作区在 `/home` 下时,旧规则会拒绝工作区自己的路径)。

## A 部分实现要点(照抄 pi)

| 清单项 | 实现 |
|---|---|
| A1 磁盘格式 | `SKILL.md` frontmatter 契约(`name`/`description`/`disable-model-invocation`) |
| A2 数据模型 | `Skill` frozen dataclass:`name/description/path/base_dir/source/disable_model_invocation` |
| A3 发现规则 | 含 `SKILL.md` 的目录即 skill 根、不再下钻;尊重 `.gitignore/.ignore/.fdignore`;跳过点目录与 `__pycache__/.venv/.git/node_modules`;symlink 用 realpath 去重;名字冲突先到先得、后者仅 warning |
| A4 校验 | name `[a-z0-9-]+` ≤64、无头尾 `-`、无连续 `--`;description 必填 ≤1024;校验失败 warning 而非 fatal;缺 description 直接跳过;frontmatter 解析失败 warning |
| A5 渐进披露 | system prompt 只放 `name + description + location`;`disable-model-invocation=true` 不进 prompt;`/skill:name args` 展开成 `<skill>` 块 |

## B 部分实现要点(自研)

```
draft ──验证通过──▶ candidate ──人工激活──▶ active ──退化/闲置──▶ retired
   └─验证失败→ 丢弃        (候选)              (进 available_skills)   (退出池)
```

| 清单项 | 实现 |
|---|---|
| B1 状态机 | 四态用**目录**承载:`~/.pirl/skills/{active,candidate,retired}`;激活 = 移动文件 |
| B2 注册表 | `SkillRecord` dataclass + 侧车 `skill-registry.json`(status/version/retrievals/adoptions/failures/reviewer…) |
| B3 触发源 | `should_propose(trace, outcome)`:`failure` 复盘 / 同一工具序列重复 ≥3 / 显式 `/skill-extract` |
| B4 提案 | 生成 skill = 生成目录:`SKILL.md` + `scripts/` + `tests/case_N.{in,out}`(3~5 留出样例) |
| B5 验证门 | `verify(proposal)` 在**留出样例**上独立跑 `scripts/run`,不重放成功轨迹;失败丢弃并记原因 |
| B6 闭环 | `on_run_end` → 提案 → 验证 → candidate(或丢弃);`activate` 人工;`on_retrieve/on_adopt` 计数;`periodic_retire` 退役 |

## 运行

```bash
# 依赖仅 PyYAML
pip install -r requirements.txt

# 测试
python -m pytest -q          # 192 passed

# ---- Part A ----
python -m pirl scan <dir>                 # 发现 skill
python -m pirl prompt <dir>               # 生成渐进披露 XML
python -m pirl expand <dir> pdf-tools convert a.pdf   # /skill 展开

# ---- Part B ----
python -m pirl init [ROOT]                # 初始化 store(默认 ~/.pirl/skills)
python -m pirl ls [ROOT] --status active  # 查看注册表
python -m pirl activate <name> --reviewer blanca --reason "..."  # 人工激活
python -m pirl retire <name> --reason "..."                      # 退役
python -m pirl verify <dir>               # 跑一个提案目录的留出样例
```

## 编程接口

```python
from pirl import (
    # Part A
    load_skills, format_skills_for_prompt, expand_skill_command,
    # Part B
    SkillStore, SelfGenLoop, Proposal, CaseSpec, should_propose, verify,
)

# ---- Part A ----
result = load_skills(cwd=".", agent_dir="~/.pirl")
prompt = format_skills_for_prompt(result.skills, file_read_tool="read")
expanded = expand_skill_command("/skill:pdf-tools convert a.pdf", result.skills)

# ---- Part B: 生命周期 ----
store = SkillStore("~/.pirl/skills")     # active/candidate/retired + registry
store.activate("pdf-tools", reviewer="blanca", reason="approved")
store.retire("pdf-tools", reason="no longer used")
store.on_retrieve("pdf-tools")           # 召回计数 +1
store.on_adopt("pdf-tools", True)        # 记录一次采用(成功)
store.periodic_retire(stale_weeks=4)     # 定期退役

# ---- Part B: 自生成闭环 ----
def generate_skill(trace, outcome) -> Proposal:   # 模型调用(你来实现)
    return Proposal(
        name="upper-text",
        description="Uppercase text. Use when converting text to uppercase.",
        body="# Upper\nRun scripts/run with the text on stdin.",
        scripts={"run": "#!/usr/bin/env python3\nimport sys\nprint(sys.stdin.read().upper(), end='')\n"},
        tests=[CaseSpec("hello\n", "HELLO\n"), CaseSpec("x\n", "X\n")],
    )

loop = SelfGenLoop(store, generator=generate_skill)
res = loop.on_run_end(trace=[], outcome="failure")
print(res.status)   # "candidate" | "rejected" | "skipped" | "error"
```

## 真实模型接入(D 之后)

把自生成闭环接上真实模型调用:`llm.py` 提供零依赖的 OpenAI 兼容
`/chat/completions` 客户端(`urllib`,不加 HTTP 依赖),`generate.py` 把
`loop.py` 里留作 stub 的 `generate_skill` 补上——让模型把 run trace 变成
带留出测试的 `Proposal`。

```python
from pirl import ChatClient, client_from_env, SelfGenLoop, SkillStore, generator_from_client

client = client_from_env()                  # PIRL_BASE_URL / PIRL_API_KEY / PIRL_MODEL
store = SkillStore("~/.pirl/skills")
loop = SelfGenLoop(store, generator=generator_from_client(client))

res = loop.on_run_end(trace=[...], outcome="failure")
# 触发 → 模型提案(带 tests)→ 留出样例验证 → candidate / rejected
print(res.status, res.name, res.message)
```

## Agent 运行时(D 之后第 2 片)

`tools.py` 提供工作区受限的文件工具和一个执行工具 `bash`(路径逃逸会被拒绝,输出截断),`agent.py`
把 Part A(skills 进系统提示)和 Part B(末尾自生成)接在一个 turn loop 上。
循环结束条件:模型给出纯文本答复(视为 success),或调用 `finish` 工具
(`success` true/false)。skill 效用统计已自动接线:模型 `read_file` 命中 active
skill 路径 → `on_retrieve`,运行结束按 outcome → `on_adopt`。

```bash
export PIRL_BASE_URL=https://api.deepseek.com/v1
export PIRL_API_KEY=...
export PIRL_MODEL=deepseek-chat

# bash is the only execution tool and is on by default (direct host backend:
# cwd is the workspace, no OS-level isolation).
python -m pirl run "fix the failing test in tests/" --cwd /path/to/repo

# Turn execution off entirely (the model can still read, search, and edit files).
python -m pirl run "review the implementation" --cwd /path/to/repo --no-bash
python -m pirl run "review the implementation" --cwd /path/to/repo --executor none

默认不限制 step 数；使用总 token、运行时间、工具调用数、费用和停滞检测控制运行边界。需要兼容旧行为时仍可传 `--max-steps N`。

# Optional Rust execution kernel (build rust-exec first):
# export PIRL_EXECUTOR=/path/to/pirl-exec
# python -m pirl run "fix tests" --executor rust --approval auto
```

### 可插拔执行层

模型只调用一个执行工具:`bash({"command": "..."})`。`exec({"argv": [...]})` 仍是
宿主内部通道(验证器、第三方插件、`--executor` 的 argv 入口),但不再出现在模型的工具表里。
选中 `rust`/`none` 后端时,`bash` 会被包装成 `bash -c <command>` 走同一个执行器,所以
换后端/禁用执行对模型是真实生效的。执行位置和外围控制由
`Executor` 插件决定；内置 `direct` 直接使用宿主用户权限，`none` 禁止进程执行，
设置 `PIRL_EXECUTOR` 后还可选择 `rust`。`PolicyExecutor`、`ApprovalExecutor` 和
`AuditExecutor` 是可组合中间件。启动时会输出 backend、隔离声明、文件系统/网络
边界和审批模式，这些信息也写入 run record。`direct` 明确不提供隔离。

第三方包可以通过 Python entry point 注册执行器：

```toml
[project.entry-points."pirl.executors"]
docker = "my_pirl_plugin:create_executor"
```

factory 接收已解析的 workspace `Path` 并返回实现 `Executor` 协议的对象。只有用户
通过 `--executor docker` 显式选择时才加载该插件；pirl 本体不会自动导入所有已安装
执行器。容器、VM、SSH 等后端因此可以放在外围实现，而无需修改 Agent loop 或工具
协议。

```python
from pirl import Agent, ChatClient, SelfGenLoop, SkillStore, ToolRegistry, builtin_tools, generator_from_client

client = ChatClient(base_url=..., api_key=..., model=...)
store = SkillStore()
registry = ToolRegistry(".", builtin_tools())
loop = SelfGenLoop(store, generator=generator_from_client(client))
agent = Agent(client, skills=store.load_active_skills().skills, tools=registry, self_gen=loop, skill_store=store)
result = agent.run("summarize this repo")
# result.answer / result.outcome / result.trace / result.loop_result
```

## Harness 与 Agent 内核

`Agent` 负责模型-工具循环、完成协议、取消、压缩和 checkpoint 回调。任务生命周期由 `pirl.harness.TaskHarness` 负责 workspace snapshot、run memory 和 `RunRecord` 产物。`TaskSpec` 是 CLI、benchmark runner 和未来 replay driver 共用的任务边界。

`pirl.protocols` 提供 `ModelProvider`、`ToolProvider`、`Verifier`、`EventSink` 和 `CheckpointWriter` 这些结构化接口。skill、self-generation、memory、git safety 和 eval 属于可注入的外围能力，不是 Agent kernel 的完成判定。

## 与 pi 的差异(有意为之)

- 依赖:**只用 PyYAML**;gitignore 匹配器自实现(注释/取反/锚定/目录限定/`*`/`?`/`**`/字符类)。
- `node_modules` 之外追加跳过 `__pycache__`/`.venv`/`.git`/`.pytest_cache`。
- 配置目录名用 `.pirl`(pi 用 `.pi`)。
- **不支持 pi 的"独立 Markdown skill"**(根目录下单个 `.md` 文件直接当 skill):只认目录 + `SKILL.md`。这是有意的简化,目录形式本就是推荐的可移植形式。
- **B 部分是 pi 没有的新增**:四态生命周期、验证门(留出样例)、效用注册表、退役机制、触发源。

## 关键风险护栏(已内建)

- **skill 噪声爆炸** → `periodic_retire` 退役/裁剪。
- **自证循环** → 验证门只跑留出样例 `tests/`,不回放成功轨迹。
- **上下文膨胀** → 渐进式披露(正文按需加载)。
- **漂移** → `Proposal.from_dir` + `verify` 支持定期重验证。
- **奖励黑客** → 高风险 skill 永远走 `activate` 人工审核(`candidate → active` 无自动通道)。

## 集成约定(接入 agent loop 时)

`Agent` 已经自动接线:模型 `read_file` 命中 active skill 路径时调
`store.on_retrieve(name)`,运行结束按 outcome 调 `store.on_adopt(name, success)`。
只有在你**自己写 loop、不用 `Agent`** 时才需要手动在这两个点调用。

字段口径:`adoptions` = 总采用次数,`failures` = 其中失败的子集,
`success_rate = (adoptions - failures) / adoptions`。

## 已知边界

- **单进程假设**:registry 是读-改-写,无进程间锁;写用临时文件 + `os.replace`
  原子替换,但并发进程同时更新会丢写。多进程场景需自行加锁。
- **draft 记录是审计轨迹**:`reject()` 会为被丢弃的提案留一条 `status="draft"`
  记录,`periodic_retire` 不清理它;长期运行会累积,可按需手动删除。
