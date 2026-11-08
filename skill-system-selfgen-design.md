# Python 版 skill 系统 + 自生成闭环 设计清单

> 来源:精读 pi(earendil-works/pi)源码后的可移植设计提炼。
> 语言:Python。目标:重写一个本地 coding agent,以 skill 为自生成单位,支持按需专门化。

## 总览

两部分:

- **A. skill 一等公民** —— 照抄 pi 的设计(语言无关)。
- **B. 自生成闭环** —— pi 没有,需要自己设计(这是"大方向一"的创新点)。

---

## A. Skill 作为一等公民(照抄 pi)

### A1. 磁盘格式

```
<skill-name>/
├── SKILL.md          # frontmatter + Markdown 正文
├── scripts/          # 可选,可执行脚本
├── references/       # 可选,参考文档
└── assets/           # 可选,模板/资源
```

`SKILL.md` 的 frontmatter 契约(Agent Skills 规范):

```yaml
---
name: pdf-tools
description: Extract text/tables from PDF. Use when reading, converting, inspecting PDFs.
disable-model-invocation: false   # true = 只能 /skill:name 显式调用
---
# 正文:直接指令
```

### A2. 数据模型

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path          # SKILL.md 绝对路径
    base_dir: Path      # SKILL.md 父目录,相对路径解析基准
    source: str         # "user" | "project" | "path"
    disable_model_invocation: bool = False
```

### A3. 发现规则

1. 递归扫描;**含 `SKILL.md` 的目录是 skill 根,不再往里递归**。
2. 尊重 `.gitignore` / `.ignore` / `.fdignore`;Python 版额外跳过 `__pycache__`、`.git`、`.venv`、`node_modules`。
3. 跳过点开头目录;跟随 symlink 但用 `realpath` 去重。
4. **名字冲突:先发现的赢,后者发 warning,不覆盖。**

### A4. 校验(放宽为 warning)

- `name`:`[a-z0-9-]+`,≤64,不以 `-` 开头/结尾,无连续 `--`。
- `description`:必填,≤1024。
- 校验失败 → **warning 而非 fatal**;**缺 description → 直接跳过不加载**。
- frontmatter 解析失败 → warning。

### A5. 渐进式披露(核心机制)

system prompt 里**只放 name + description + location**,不放正文:

```xml
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
Resolve relative paths against the skill's directory (parent of SKILL.md).

<available_skills>
  <skill>
    <name>pdf-tools</name>
    <description>Extract text/tables from PDF. Use when...</description>
    <location>/home/u/.pirl/skills/active/pdf-tools/SKILL.md</location>
  </skill>
</available_skills>
```

正文按需加载,两种路径:

- 模型自己用 read 工具读 `SKILL.md`(pi 默认 `fileReadTool="read"`)。
- `/skill:name args` 命令展开成:

```xml
<skill name="pdf-tools" location="...">
References are relative to /home/u/.pirl/skills/active/pdf-tools.

<正文>
</skill>
<args>
```

> 这是"无限扩展各种专用 skill 而不撑爆上下文"的唯一前提,必须做对。

---

## B. 自生成闭环(pi 没有,自己设计)

### B1. 生命周期状态机

```
              ┌── revise ──┐
              ↓            │
 draft ──验证通过──▶ candidate ──人工激活──▶ active ──退化/闲置──▶ retired
   └─验证失败→ 丢弃        (候选)              (进 available_skills)   (退出池)
```

四个状态,用**目录**承载(激活 = 移动文件,最简单且可审):

```
~/.pirl/skills/
├── active/       # 只有这里进 <available_skills>
├── candidate/    # 待人工激活
└── retired/      # 已退役,不加载
```

### B2. 生命周期注册表

skill 文件保持不可变,状态与统计放侧车注册表(`skill-registry.json` 或 SQLite):

```python
@dataclass
class SkillRecord:
    name: str
    status: str            # draft | candidate | active | retired
    created_at: str
    version: int           # 每次 revise +1
    # 效用统计(MemRL 的 Q 值思想)
    retrievals: int = 0    # 被召回进 prompt 的次数
    adoptions: int = 0     # 总采用次数(成功 + 失败)
    failures: int = 0      # 其中采用后任务失败的子集
    # success_rate = (adoptions - failures) / adoptions
    # 审核
    reviewer: str = ""
    activate_reason: str = ""
```

### B3. 触发源(何时提案一个 skill)

```python
def should_propose(trace, outcome) -> bool:
    return (
        outcome == FAILURE                     # 失败复盘
        or repeated_tool_pattern(trace, n=3)   # 同一工具序列重复 >= 3 次
        or user_requested_skill_extraction()   # 显式 /skill-extract
    )
```

### B4. 提案结构(生成的 skill 必须带测试)

生成一个 skill = 生成一个目录:

```
candidate/<name>/
├── SKILL.md          # description 当路由词,正文当指令
├── scripts/          # 若有脚本
└── tests/            # 关键:3~5 个 (input, expected) 留出样例
    └── case_1.{in,out}
```

**为什么必须带 tests**:验证门要用留出样例,防止"在刚成功的轨迹上自证"。

### B5. 验证门(区分 draft 和 candidate 的关卡)

```python
def verify(proposal) -> bool:
    # 在留出样例上独立跑,而不是重放成功轨迹
    return all(run_case(c) == c.expected for c in proposal.tests)
```

- 验证通过 → `candidate/`,提示用户"新候选 skill:name,请审"。
- 验证失败 → 丢弃,记失败原因,不进入候选。
- `candidate → active` 必须**人工** `/skill activate <name> --reason "..."`(高风险 skill 永不自动激活)。

### B6. 闭环伪代码

```python
def on_run_end(trace, outcome):
    if should_propose(trace, outcome):
        p = generate_skill(trace)          # 模型调用 → SKILL.md + tests/
        if verify(p):
            move_to(p, "candidate/")
            notify(f"新候选 skill「{p.name}」,待审核")
        else:
            log(f"skill {p.name} 验证失败,丢弃")

def activate(name, reviewer, reason):      # 人工
    move(f"candidate/{name}", f"active/{name}")
    registry[name] = SkillRecord(status="active", reviewer=reviewer, ...)

def on_retrieve(name): registry[name].retrievals += 1
def on_adopt(name, success):
    registry[name].adoptions += 1
    if not success: registry[name].failures += 1

def periodic_retire():
    for s in active_skills():
        if s.retrievals == 0 and stale(s, weeks=4):
            retire(s)   # active/ -> retired/
        elif s.success_rate < 0.3 and s.adoptions >= 5:
            retire(s)
```

---

## C. 抄什么 / 砍什么 / 加什么

| | 内容 |
|---|---|
| **照抄 pi** | name/description 契约、校验规则、渐进披露 XML 格式、`/skill:name` 展开、gitignore 尊重、冲突处理、symlink 去重 |
| **砍掉(JS 专属)** | jiti loader、`node_modules` 跳过 → 换成 `__pycache__`/`.venv`/`.git` |
| **有意不抄** | pi 的"独立 Markdown skill"(根目录单个 `.md` 当 skill);只认目录 + `SKILL.md` |
| **pi 没有、你要加** | 生命周期四态 + 目录承载、验证门(tests/ 留出样例)、效用注册表(召回/采用/成功率)、退役机制、触发源 |

---

## D. 落地顺序(每步可独立验收)

1. **Skill 模型 + 发现 + 校验**(A2/A3/A4):`Skill` dataclass、frontmatter 解析(用 `yaml`)、递归发现、name 冲突。验收:一个含 3 个 skill 的目录扫出正确结果。
2. **渐进披露**(A5):system prompt builder 输出 `<available_skills>`。验收:prompt 里只有 name/description/location,无正文。
3. **命令展开 + read 路径**(A5):`/skill:name` 展开、read 工具能读 SKILL.md。
4. **生命周期注册表**(B1/B2):active/candidate/retired 目录 + registry JSON。验收:activate/retire 能正确移动文件并更新 registry。
5. **自生成闭环**(B3-B6):触发 → 提案(带 tests)→ 验证 → candidate → 人工 activate。验收:一个失败任务能产出带测试的候选 skill,留出样例验证能拦掉一个假 skill。
6. **效用与退役**(B6 后半):召回/采用计数 + 定期退役。验收:一个"从不被召回"的 skill 4 周后进 retired。

---

## 关键风险与护栏

- **skill 噪声爆炸**:库无限膨胀 → 检索退化 → 每次召错 skill。→ 必须带退役/裁剪。
- **自证循环**:在成功轨迹上验证自己的 skill。→ 必须留出样例。
- **上下文膨胀**:skill 正文泄漏进每个请求。→ 渐进式披露。
- **漂移**:skill 随代码库演化而失效。→ skill 带测试、定期重验证。
- **奖励黑客**:agent 学会"标记自己成功"而非"真的做好"。→ 高风险 skill 永远人审。

## 集成约定与已知边界

- **效用统计是手动接线**:接入 agent loop 时,必须在"召回进 prompt"和"采用后成功/失败"两个点分别调 `on_retrieve` / `on_adopt`;否则 `periodic_retire` 会因 `retrievals == 0` 误退役活跃 skill。
- **单进程假设**:registry 无进程间锁,并发更新会丢写;多进程需自行加锁。
- **draft 记录是审计轨迹**:`reject()` 留 `status="draft"` 记录且不被 `periodic_retire` 清理,长期运行会累积。

---

## 参考源码(pi,本地 clone)

- `/home/blanca/pi-reference/packages/coding-agent/src/core/skills.ts` —— 发现 + 校验 + 渐进披露(505 行,第 1~4 步直接对应)。
- `/home/blanca/pi-reference/packages/coding-agent/src/core/agent-session.ts` —— `/skill:name` 展开(约 2295 行)。
- `/home/blanca/pi-reference/packages/coding-agent/src/utils/frontmatter.ts` —— frontmatter 解析。
- `/home/blanca/pi-reference/examples/extensions/subagent/` —— 专门 agent(子进程 + `.md` 定义),对应后续的专门化方向。
