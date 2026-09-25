"""Shared profiler invocation helpers.

Two loops drive the Profiler agent today: ``agent/loop.py`` (per-round
profiling, owns the round/progress.md side-effects) and
``evolve/loop.py`` (per-offspring profiling, with an optional
Pareto-frontier addendum).  Both build an MCP server spec for the
analysis tools (torch profiler or nsys), render their own system
prompt, and call ``ctx.invoke(kind="profiler", ...)`` with a
``ProfilerSummary`` fallback.

This module owns the parts that are identical across the two: the
``MCPServerSpec`` factory, the agent-invocation wrapper, and the
advisory custom-profiler path (run a bundle-declared command, then use
its structured summary directly or ask the profiler agent to interpret
the capture). Each loop still renders its own prompt (the templates and
bound variables differ) and decides what to do with the returned summary.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic import ValidationError

from vibesys.events import FrameworkSource
from vibesys.loops.gates import framework_command_timeout
from vibesys.profilers import (
    ProfilerDefinition,
    ProfilerKind,
    profiler_definition,
    require_profiler_kind,
)
from vibesys.render.sink import output_sink
from vibesys.schemas import ProfilerSummary
from vs_agent.api import MCPServerSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vibesys.run import LoopContext

    ProfilerInvoke = Callable[..., ProfilerSummary]

# Environment variable that tells a custom profiler command where to write
# durable artifacts (workspace-relative path, created before the command runs).
PROFILE_DIR_ENV_VAR = "VIBESYS_PROFILE_DIR"
# Structured-summary file a custom profiler may write under $VIBESYS_PROFILE_DIR.
CUSTOM_PROFILER_SUMMARY_FILE = "summary.json"
# Prompt used when the profiler agent interprets a custom command's capture.
CUSTOM_PROFILER_TEMPLATE = "profilers/custom.j2"
# Retained tail of a custom profiler's combined output. The capture and the
# interpretation prompt share this bound so the agent sees what was recorded.
CUSTOM_PROFILER_OUTPUT_TAIL_CHARS = 8000
# Exit code every sandbox reports when a command exceeds its timeout.
TIMEOUT_EXIT_CODE = 124

_PROFILER_USER_PROMPT = (
    "Profile the server and return exactly one JSON object matching the schema above."
)
_INTERPRET_USER_PROMPT = (
    "Interpret the captured profiler evidence above and return exactly one JSON "
    "object matching the schema."
)


def profiler_fallback(suggestions: str) -> ProfilerSummary:
    """The summary recorded when the profiler agent returns nothing structured."""
    return ProfilerSummary(
        analysis="Profiler produced no structured response.",
        bottlenecks="n/a",
        suggestions=suggestions,
        perf_metric=None,
        perf_unit=None,
    )


def effective_profiler_definition(
    profiler_kind: ProfilerKind,
    *,
    supports_torch_profiler: bool = False,
) -> ProfilerDefinition:
    """Return the already-resolved profiler declaration.

    Context creation resolves the requested profiler against both the domain
    and the run environment's declared capabilities.  Do not perform a second
    interface-based substitution here: it can replace a supported remote
    capture path with a profiler that the environment cannot execute.
    """
    kind = require_profiler_kind(profiler_kind)
    if kind is ProfilerKind.NONE:
        message = "No profiler prompt exists when profiling is disabled."
        raise ValueError(message)
    definition = profiler_definition(kind)
    if definition.requires_domain_torch_support and not supports_torch_profiler:
        message = "The selected domain does not provide Torch profiler support."
        raise ValueError(message)
    return definition


def warn_profiler_failed(exc: Exception, *, round_label: str) -> None:
    """Report a profiler-agent failure as a non-fatal framework warning."""
    output_sink().framework_warning(
        "profiler failed",
        detail=str(exc),
        source=FrameworkSource.LOOP,
        round_label=round_label,
    )


def mcp_spec(profiler_kind: ProfilerKind) -> MCPServerSpec | None:
    """Build an ``MCPServerSpec`` that spawns the analysis MCP server.

    Returns ``None`` for :attr:`ProfilerKind.NONE`, which callers treat as
    "skip MCP": the profiler agent still runs, just without tool access.
    """
    kind = require_profiler_kind(profiler_kind)
    if kind is ProfilerKind.NONE:
        return None
    definition = profiler_definition(kind)
    return MCPServerSpec(
        name=definition.mcp_name,
        command="python",
        args=(definition.server_path,),
    )


@dataclass(frozen=True)
class ProfilerTurn:
    """One Profiler agent invocation: the rendered prompt plus its bookkeeping."""

    system_prompt: str
    round_label: str
    fallback_suggestions: str = "Re-run profiling on the next round."
    user_prompt: str = _PROFILER_USER_PROMPT


def invoke_profiler_agent(
    ctx: LoopContext,
    turn: ProfilerTurn,
    *,
    mcp_servers: list[MCPServerSpec] | None = None,
    invoke: ProfilerInvoke | None = None,
) -> ProfilerSummary:
    """Run the Profiler agent once and return its :class:`ProfilerSummary`.

    The invocation core shared by every profiler path. ``invoke`` defaults to
    ``ctx.invoke``; the agent loop passes its read-only role wrapper instead.
    Exceptions propagate: callers decide whether a failure is fatal.
    """
    invoke_fn = ctx.invoke if invoke is None else invoke
    return invoke_fn(
        kind="profiler",
        system_prompt=turn.system_prompt,
        user_prompt=turn.user_prompt,
        response_cls=ProfilerSummary,
        fallback_factory=lambda: profiler_fallback(turn.fallback_suggestions),
        round_label=turn.round_label,
        mcp_servers=mcp_servers,
    )


def _guarded_profiler_agent(
    ctx: LoopContext,
    turn: ProfilerTurn,
    *,
    mcp_servers: list[MCPServerSpec] | None,
    invoke: ProfilerInvoke | None,
) -> ProfilerSummary | None:
    """Run the agent core; turn any failure into a framework warning and ``None``."""
    try:
        return invoke_profiler_agent(ctx, turn, mcp_servers=mcp_servers, invoke=invoke)
    except Exception as exc:  # noqa: BLE001  # lint-waiver: LW-010265 [BLE001]; configured profiler failures become framework warnings so the run continues without profile data.
        warn_profiler_failed(exc, round_label=turn.round_label)
        return None


def invoke_profiler(
    ctx: LoopContext,
    *,
    system_prompt: str,
    round_label: str,
    fallback_suggestions: str = "Re-run profiling on the next round.",
) -> ProfilerSummary | None:
    """Run the built-in Profiler agent and return its :class:`ProfilerSummary`.

    Side-effect free: the caller owns logging the result, writing it to
    progress.md, snapshotting the workspace, etc. Returns ``None`` when
    profiling is disabled or the agent raised (the caller decides whether
    that's fatal).
    """
    if ctx.profiler_kind is ProfilerKind.NONE:
        return None
    spec = mcp_spec(ctx.profiler_kind)
    turn = ProfilerTurn(
        system_prompt=system_prompt,
        round_label=round_label,
        fallback_suggestions=fallback_suggestions,
    )
    return _guarded_profiler_agent(
        ctx, turn, mcp_servers=[spec] if spec is not None else None, invoke=None
    )


# ---------------------------------------------------------------------------
# Custom profiler command (advisory; never fails a round)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomProfilerCapture:
    """What one run of the bundle-declared profiler command left behind."""

    exit_code: int | None
    output: str
    artifact_location: str
    artifact_files: tuple[str, ...]
    summary: ProfilerSummary | None

    @property
    def has_evidence(self) -> bool:
        """True when the command succeeded and produced output or artifact files."""
        return self.exit_code == 0 and bool(self.output.strip() or self.artifact_files)


def _parse_summary(text: str) -> ProfilerSummary | None:
    """Return the ``ProfilerSummary`` encoded by ``text``, or ``None`` when it is not one."""
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ProfilerSummary.model_validate(payload)
    except ValidationError:
        return None


def _structured_summary(artifact_root: Path, stdout: str) -> ProfilerSummary | None:
    """Prefer ``summary.json`` under the artifact root, then a JSON object on stdout."""
    summary_path = artifact_root / CUSTOM_PROFILER_SUMMARY_FILE
    if summary_path.is_file():
        summary = _parse_summary(summary_path.read_text(encoding="utf-8", errors="replace"))
        if summary is not None:
            return summary
    return _parse_summary(stdout)


def _artifact_files(artifact_root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(artifact_root).as_posix()
            for path in artifact_root.rglob("*")
            if path.is_file()
        )
    )


def run_custom_profiler(
    ctx: LoopContext,
    *,
    command: str,
    timeout_seconds: int | None,
    artifact_root: Path,
    round_label: str,
) -> CustomProfilerCapture:
    """Run the bundle-declared profiler command and collect what it captured.

    ``artifact_root`` is created on the host and exported to the command as
    ``$VIBESYS_PROFILE_DIR`` (workspace-relative). The command runs through
    the judge backend from the repository root, like the evaluator commands.
    A nonzero exit (including the timeout exit code) is reported as a
    framework warning and yields a capture with no evidence; it never raises.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_location = artifact_root.relative_to(ctx.workspace).as_posix()
    shell_command = f"env {PROFILE_DIR_ENV_VAR}={shlex.quote(artifact_location)} {command}"
    timeout = framework_command_timeout(ctx, timeout_seconds)
    if timeout is None:
        result = ctx.judge_backend.execute(shell_command)
    else:
        result = ctx.judge_backend.execute(shell_command, timeout=timeout)
    output = result.output[-CUSTOM_PROFILER_OUTPUT_TAIL_CHARS:]
    artifact_files = _artifact_files(artifact_root)
    if result.exit_code != 0:
        reason = (
            "timed out"
            if result.exit_code == TIMEOUT_EXIT_CODE
            else f"exit code {result.exit_code}"
        )
        output_sink().framework_warning(
            "custom profiler failed",
            detail=f"{reason}: {command}\n{output}".rstrip(),
            source=FrameworkSource.LOOP,
            round_label=round_label,
        )
        return CustomProfilerCapture(
            exit_code=result.exit_code,
            output=output,
            artifact_location=artifact_location,
            artifact_files=artifact_files,
            summary=None,
        )
    return CustomProfilerCapture(
        exit_code=result.exit_code,
        output=output,
        artifact_location=artifact_location,
        artifact_files=artifact_files,
        summary=_structured_summary(artifact_root, result.stdout or result.output),
    )


def custom_profiler_prompt_context(capture: CustomProfilerCapture) -> dict[str, object]:
    """The extra variables ``profilers/custom.j2`` reads beyond the shared profiler kwargs."""
    return {
        "captured_output": capture.output,
        "captured_output_limit": CUSTOM_PROFILER_OUTPUT_TAIL_CHARS,
        "artifact_files": capture.artifact_files,
        "artifact_location": capture.artifact_location,
    }


def interpret_custom_profiler(
    ctx: LoopContext,
    capture: CustomProfilerCapture,
    turn: ProfilerTurn,
    *,
    invoke: ProfilerInvoke | None = None,
) -> ProfilerSummary | None:
    """Turn a custom profiler capture into a :class:`ProfilerSummary`.

    A structured summary the command emitted is used as is. Otherwise the
    profiler agent reads the capture (no MCP tools) when there is evidence to
    read; a failed or empty capture yields ``None`` without an agent call.
    Agent failures become framework warnings and ``None``.
    """
    if capture.summary is not None:
        return capture.summary
    if not capture.has_evidence:
        return None
    return _guarded_profiler_agent(
        ctx,
        replace(turn, user_prompt=_INTERPRET_USER_PROMPT),
        mcp_servers=None,
        invoke=invoke,
    )
