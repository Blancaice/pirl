"""Self-generation closed loop (B6).

Ties the pieces together::

    def on_run_end(trace, outcome):
        if should_propose(trace, outcome):
            p = generate_skill(trace)          # model call -> SKILL.md + tests/
            if verify(p):
                move_to(p, "candidate/")
                notify(f"新候选 skill「{p.name}」,待审核")
            else:
                log(f"skill {p.name} 验证失败,丢弃")

``generate_skill`` is a model call we cannot perform ourselves, so it is injected
as a ``SkillGenerator`` callable (or a ready-made :class:`Proposal` is passed in).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .lifecycle import SkillRecord, SkillStore
from .proposal import (
    Proposal,
    VerificationResult,
    should_propose,
    verify,
)

# A model call that turns a run trace into a skill proposal (SKILL.md + tests/).
SkillGenerator = Callable[[Sequence, str], Proposal]


@dataclass
class LoopResult:
    """Outcome of :meth:`SelfGenLoop.on_run_end`."""

    proposed: bool  # did a trigger fire and a proposal get generated?
    status: str  # "skipped" | "candidate" | "rejected" | "error"
    name: str = ""
    verification: Optional[VerificationResult] = None
    record: Optional[SkillRecord] = None
    message: str = ""


class SelfGenLoop:
    """Runs the propose -> verify -> candidate (or discard) closed loop."""

    def __init__(
        self,
        store: SkillStore,
        *,
        generator: Optional[SkillGenerator] = None,
        n_repeat: int = 3,
    ) -> None:
        self.store = store
        self.generator = generator
        self.n_repeat = n_repeat

    def on_run_end(
        self,
        trace: Sequence,
        outcome: str,
        *,
        proposal: Optional[Proposal] = None,
        generator: Optional[SkillGenerator] = None,
        user_requested: bool = False,
    ) -> LoopResult:
        """Called at the end of a run. Proposes and verifies a skill if warranted."""
        if not should_propose(
            trace, outcome, user_requested=user_requested, n=self.n_repeat
        ):
            return LoopResult(False, "skipped", message="no trigger fired")

        if proposal is None:
            gen = generator or self.generator
            if gen is None:
                return LoopResult(
                    True, "error", message="trigger fired but no proposal/generator given"
                )
            try:
                proposal = gen(trace, outcome)
            except Exception as exc:  # noqa: BLE001 - model output is untrusted
                return LoopResult(
                    True, "error", message=f"generator raised: {exc}"
                )

        if not isinstance(proposal, Proposal):
            return LoopResult(
                True, "error", message="generator must return a Proposal"
            )

        # Materialize into a transient draft dir, then verify on held-out cases.
        draft = Path(tempfile.mkdtemp(prefix="pirl-draft-"))
        try:
            result = verify(proposal, work_dir=draft)
        except Exception as exc:  # noqa: BLE001 - surface as an error result
            shutil.rmtree(draft, ignore_errors=True)
            return LoopResult(
                True, "error", name=proposal.name, message=f"verify raised: {exc}"
            )

        if result.passed:
            try:
                record = self.store.promote_to_candidate(proposal, draft)
            except Exception as exc:  # noqa: BLE001
                shutil.rmtree(draft, ignore_errors=True)
                return LoopResult(
                    True, "error", name=proposal.name, message=f"promote failed: {exc}"
                )
            # draft was moved into candidate/ by promote_to_candidate.
            return LoopResult(
                True,
                "candidate",
                name=proposal.name,
                verification=result,
                record=record,
                message=f"新候选 skill「{proposal.name}」,待审核",
            )

        # Verification failed -> discard, keep an audit record of the reason.
        shutil.rmtree(draft, ignore_errors=True)
        self.store.reject(proposal.name, result.reason)
        return LoopResult(
            True,
            "rejected",
            name=proposal.name,
            verification=result,
            message=f"skill {proposal.name} 验证失败,丢弃",
        )

    # ------------------------------------------------------------------ #
    # Human-in-the-loop + utility bookkeeping (delegated to the store)
    # ------------------------------------------------------------------ #

    def activate(self, name: str, reviewer: str = "", reason: str = "") -> SkillRecord:
        """Human activation: candidate -> active."""
        return self.store.activate(name, reviewer=reviewer, reason=reason)

    def retire(self, name: str, reason: str = "") -> SkillRecord:
        return self.store.retire(name, reason=reason)

    def on_retrieve(self, name: str) -> Optional[SkillRecord]:
        return self.store.on_retrieve(name)

    def on_adopt(self, name: str, success: bool) -> Optional[SkillRecord]:
        return self.store.on_adopt(name, success)

    def periodic_retire(self, **kwargs):
        return self.store.periodic_retire(**kwargs)
