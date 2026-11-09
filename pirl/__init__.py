"""pirl — a Python port of the skill system.

Two parts:

* **Part A** makes skills first-class citizens, ported from pi
  (``packages/coding-agent/src/core/skills.ts``):
  A1 disk format, A2 data model, A3 discovery, A4 validation, A5 progressive
  disclosure.
* **Part B** is the self-generation closed loop (pi does not have this):
  B1 lifecycle state machine, B2 sidecar registry, B3 trigger sources,
  B4 proposal structure, B5 verification gate, B6 the loop + utility/retirement.

Public API::

    from pirl import (
        load_skills, format_skills_for_prompt, expand_skill_command,   # Part A
        SkillStore, SelfGenLoop, Proposal, should_propose, verify,     # Part B
    )
"""

from .frontmatter import parse_frontmatter, strip_frontmatter
from .prompt import PromptSection, format_skills_for_prompt, render_sections
from .skills import (
    Diagnostic,
    LoadSkillsResult,
    Skill,
    discover_skills,
    load_skill_from_file,
    load_skills,
    validate_description,
    validate_name,
)
from .lifecycle import (
    SkillRecord,
    SkillStore,
    default_store_root,
)
from .proposal import (
    CaseResult,
    Proposal,
    CaseSpec,
    ToolCall,
    VerificationResult,
    repeated_tool_pattern,
    should_propose,
    verify,
)
from .loop import LoopResult, SelfGenLoop
from .llm import (
    ChatClient,
    Completion,
    LLMConfigError,
    LLMError,
    LLMHTTPError,
    LLMResponseError,
    ToolCall,
    ToolSpec,
    client_from_env,
)
from .generate import (
    build_generation_messages,
    generator_from_client,
    parse_proposal_json,
)
from .tools import Tool, ToolRegistry, ToolResult, builtin_tools
from .agent import Agent, FinishCalled, RunResult, StepStart, ToolCalled
from .run_memory import memory_path, read_memory, write_memory
from .env import load_dotenv
from . import ui

__all__ = [
    # UI
    "ui",
    # Part A
    "Diagnostic",
    "LoadSkillsResult",
    "PromptSection",
    "Skill",
    "render_sections",
    "discover_skills",
    "expand_skill_command",
    "format_skills_for_prompt",
    "load_skill_from_file",
    "load_skills",
    "parse_frontmatter",
    "strip_frontmatter",
    "validate_description",
    "validate_name",
    # Part B — lifecycle
    "SkillRecord",
    "SkillStore",
    "default_store_root",
    # Part B — proposal / verification
    "CaseResult",
    "Proposal",
    "CaseSpec",
    "ToolCall",
    "VerificationResult",
    "repeated_tool_pattern",
    "should_propose",
    "verify",
    # Part B — closed loop
    "LoopResult",
    "SelfGenLoop",
    # LLM client
    "ChatClient",
    "Completion",
    "LLMConfigError",
    "LLMError",
    "LLMHTTPError",
    "LLMResponseError",
    "ToolCall",
    "ToolSpec",
    "client_from_env",
    # Skill generation
    "build_generation_messages",
    "generator_from_client",
    "parse_proposal_json",
    # Agent runtime
    "Agent",
    "RunResult",
    "StepStart",
    "ToolCalled",
    "FinishCalled",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "builtin_tools",
    # Run memory + env
    "memory_path",
    "read_memory",
    "write_memory",
    "load_dotenv",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    # Lazy import so ``from pirl import expand_skill_command`` works without a
    # circular import at module load time.
    if name == "expand_skill_command":
        from .commands import expand_skill_command

        return expand_skill_command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
