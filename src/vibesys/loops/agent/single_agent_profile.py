"""Profile-section context for the single-agent round prompt.

``--inner-loop=single-agent`` has no profiler agent: the one implementer runs
the capture itself. The prompt therefore has to name the capture tool, which
is either a built-in profiler kind or the bundle-declared ``[profiler]``
command. A declared command is selected with ``profiler_kind == NONE``, so it
is passed as its own variable and the template checks it before the disabled
branch, mirroring ``orchestrator_pre_round_prompt.j2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibesys.loops.agent import issue_board
from vibesys.loops.profiler import effective_profiler_definition
from vibesys.profilers import ProfilerKind

if TYPE_CHECKING:
    from pathlib import Path

    from vibesys.run import LoopContext


def single_agent_profile_context(
    ctx: LoopContext,
    progress_path: Path,
    round_number: int,
    *,
    supports_torch_profiler: bool,
) -> dict[str, object]:
    """Return the ``## Profile`` template variables for one single-agent round.

    With a declared command, ``custom_profiler_artifact_location`` is the
    round's profiler artifact root (the directory the orchestrated path sets
    as ``VIBESYS_PROFILE_DIR``), workspace-relative, so the implementer can
    run the command under the same output contract. Both custom variables are
    ``None`` otherwise.
    """
    custom = ctx.custom_profiler
    custom_location = None
    if custom is not None:
        artifact_root = issue_board.profiler_artifact_root(progress_path, round_number)
        custom_location = issue_board.display_path(artifact_root, ctx.workspace).rstrip("/")
    effective_profiler = (
        effective_profiler_definition(
            ctx.profiler_kind, supports_torch_profiler=supports_torch_profiler
        )
        if ctx.profiler_kind is not ProfilerKind.NONE
        else None
    )
    return {
        "profiler_kind": ctx.profiler_kind,
        "profiler_support_name": (effective_profiler.support_name if effective_profiler else None),
        "profiler_mcp_name": (effective_profiler.mcp_name if effective_profiler else None),
        "custom_profiler_command": (custom.command if custom is not None else None),
        "custom_profiler_artifact_location": custom_location,
    }
