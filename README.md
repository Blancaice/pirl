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
└── __main__.py      # CLI(scan / prompt / expand / init / ls / activate / retire / verify)
tests/               # pytest 测试(97 个用例)
```

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
python -m pytest -q          # 97 passed

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

效用统计是**手动**接线,包内不自动调用。必须在两个点调用,否则
`periodic_retire` 会把活跃 skill 误退役(`retrievals == 0` 恒成立):

- **召回点**:skill 进入 prompt 时调 `store.on_retrieve(name)`。
- **采用点**:任务结束、skill 被实际采用后调 `store.on_adopt(name, success)`。

字段口径:`adoptions` = 总采用次数,`failures` = 其中失败的子集,
`success_rate = (adoptions - failures) / adoptions`。

## 已知边界

- **单进程假设**:registry 是读-改-写,无进程间锁;写用临时文件 + `os.replace`
  原子替换,但并发进程同时更新会丢写。多进程场景需自行加锁。
- **draft 记录是审计轨迹**:`reject()` 会为被丢弃的提案留一条 `status="draft"`
  记录,`periodic_retire` 不清理它;长期运行会累积,可按需手动删除。
