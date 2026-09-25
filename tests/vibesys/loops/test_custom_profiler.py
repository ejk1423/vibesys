"""Tests for the advisory custom profiler command path in ``vibesys.loops.profiler``.

``run_custom_profiler`` executes a bundle-declared command through the judge
backend and collects its capture; ``interpret_custom_profiler`` turns that
capture into a ``ProfilerSummary`` either directly (structured output) or via
one Profiler agent turn without MCP tools. ``custom_profiler_summary``
chains the two with a lazily rendered prompt, and ``declared_custom_profiler``
lifts the command out of an input bundle. All are exercised here with fake
backends and contexts, never a real sandbox or agent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from vibesys.evaluators.input_manifest import load_input_bundle
from vibesys.loops import profiler as profiler_module
from vibesys.loops.profiler import (
    CUSTOM_PROFILER_OUTPUT_TAIL_CHARS,
    PROFILE_DIR_ENV_VAR,
    TIMEOUT_EXIT_CODE,
    CustomProfilerCapture,
    CustomProfilerRun,
    ProfilerTurn,
    custom_profiler_prompt_context,
    custom_profiler_summary,
    declared_custom_profiler,
    interpret_custom_profiler,
    run_custom_profiler,
)
from vibesys.profilers import CustomProfilerCommand
from vibesys.schemas import ProfilerSummary
from vs_sandbox.api import SandboxExecutionResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vibesys.run import LoopContext

_SUMMARY = {
    "analysis": "decode is launch-bound",
    "bottlenecks": "1. kernel launches",
    "suggestions": "capture a CUDA graph",
    "perf_metric": 12.5,
    "perf_unit": "tok/s",
}


class _FakeBackend:
    """A judge backend that records the command and answers with a fixed result."""

    def __init__(
        self,
        result: SandboxExecutionResult,
        *,
        side_effect: Callable[[], None] | None = None,
    ) -> None:
        self.result = result
        self.side_effect = side_effect
        self.commands: list[tuple[str, int | None]] = []

    def execute(self, command: str, timeout: int | None = None) -> SandboxExecutionResult:
        self.commands.append((command, timeout))
        if self.side_effect is not None:
            self.side_effect()
        return self.result


def _ctx(workspace: Path, backend: object, **members: object) -> LoopContext:
    ctx = MagicMock()
    ctx.workspace = workspace
    ctx.judge_backend = backend
    ctx.run_environment_view = SimpleNamespace(framework_setup_timeout_seconds=0)
    # A MagicMock attribute is truthy, which would read as a declared command.
    ctx.custom_profiler = None
    for name, value in members.items():
        setattr(ctx, name, value)
    return cast("LoopContext", ctx)


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        profiler_module,
        "output_sink",
        lambda: SimpleNamespace(
            framework_warning=lambda text, **kw: warnings.append((text, kw)),
        ),
    )
    return warnings


def _run(
    tmp_path: Path,
    backend: _FakeBackend,
    *,
    timeout_seconds: int | None = 30,
) -> tuple[CustomProfilerCapture, Path]:
    artifact_root = tmp_path / "progress-artifacts" / "profiles" / "round-0001"
    ctx = _ctx(tmp_path, backend)
    capture = run_custom_profiler(
        ctx,
        command="python profiler/profile.py",
        timeout_seconds=timeout_seconds,
        artifact_root=artifact_root,
        round_label="round-1-profiler",
    )
    return capture, artifact_root


# ---------------------------------------------------------------------------
# run_custom_profiler
# ---------------------------------------------------------------------------


def test_structured_stdout_becomes_the_summary(tmp_path: Path) -> None:
    stdout = json.dumps(_SUMMARY)
    backend = _FakeBackend(SandboxExecutionResult(output=stdout, exit_code=0, stdout=stdout))

    capture, artifact_root = _run(tmp_path, backend)

    assert capture.summary == ProfilerSummary.model_validate(_SUMMARY)
    assert capture.has_evidence is True
    assert capture.exit_code == 0
    assert capture.artifact_files == ()
    assert artifact_root.is_dir(), "the artifact directory is created before the command runs"
    assert capture.artifact_location == "progress-artifacts/profiles/round-0001"


def test_command_runs_with_profile_dir_and_framework_timeout(tmp_path: Path) -> None:
    backend = _FakeBackend(SandboxExecutionResult(output="", exit_code=0))
    ctx = _ctx(tmp_path, backend)
    ctx.run_environment_view = SimpleNamespace(framework_setup_timeout_seconds=5)

    run_custom_profiler(
        ctx,
        command="python profiler/profile.py --fast",
        timeout_seconds=30,
        artifact_root=tmp_path / "profiles" / "r1",
        round_label="round-1-profiler",
    )

    [(command, timeout)] = backend.commands
    assert command == f"env {PROFILE_DIR_ENV_VAR}=profiles/r1 python profiler/profile.py --fast"
    assert timeout == 35, "the run environment's setup allowance extends the declared budget"


def test_command_runs_without_timeout_when_none_declared(tmp_path: Path) -> None:
    backend = _FakeBackend(SandboxExecutionResult(output="", exit_code=0))

    _run(tmp_path, backend, timeout_seconds=None)

    [(_, timeout)] = backend.commands
    assert timeout is None


def test_summary_json_in_artifact_dir_wins_over_stdout(tmp_path: Path) -> None:
    artifact_root = tmp_path / "progress-artifacts" / "profiles" / "round-0001"
    file_summary = {**_SUMMARY, "analysis": "from summary.json"}
    stdout_summary = {**_SUMMARY, "analysis": "from stdout"}

    def write_summary() -> None:
        (artifact_root / "summary.json").write_text(json.dumps(file_summary))
        (artifact_root / "trace").mkdir()
        (artifact_root / "trace" / "perf.data").write_bytes(b"\x00")

    stdout = json.dumps(stdout_summary)
    backend = _FakeBackend(
        SandboxExecutionResult(output=stdout, exit_code=0, stdout=stdout),
        side_effect=write_summary,
    )

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is not None
    assert capture.summary.analysis == "from summary.json"
    assert capture.artifact_files == ("summary.json", "trace/perf.data")


def test_invalid_summary_json_falls_back_to_stdout(tmp_path: Path) -> None:
    artifact_root = tmp_path / "progress-artifacts" / "profiles" / "round-0001"

    def write_garbage() -> None:
        (artifact_root / "summary.json").write_text('{"analysis": "missing fields"}')

    stdout = json.dumps(_SUMMARY)
    backend = _FakeBackend(
        SandboxExecutionResult(output=stdout, exit_code=0, stdout=stdout),
        side_effect=write_garbage,
    )

    capture, _ = _run(tmp_path, backend)

    assert capture.summary == ProfilerSummary.model_validate(_SUMMARY)


def test_unstructured_stdout_is_evidence_without_a_summary(tmp_path: Path) -> None:
    text = "perf stat: 1,234 cycles\n"
    backend = _FakeBackend(SandboxExecutionResult(output=text, exit_code=0, stdout=text))

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is None
    assert capture.has_evidence is True
    assert capture.output == text


def test_json_array_on_stdout_is_unstructured(tmp_path: Path) -> None:
    text = json.dumps([1, 2, 3])
    backend = _FakeBackend(SandboxExecutionResult(output=text, exit_code=0, stdout=text))

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is None
    assert capture.has_evidence is True


def test_output_is_bounded_to_the_named_tail(tmp_path: Path) -> None:
    text = "x" * (CUSTOM_PROFILER_OUTPUT_TAIL_CHARS + 100) + "tail"
    backend = _FakeBackend(SandboxExecutionResult(output=text, exit_code=0, stdout=text))

    capture, _ = _run(tmp_path, backend)

    assert len(capture.output) == CUSTOM_PROFILER_OUTPUT_TAIL_CHARS
    assert capture.output.endswith("tail")


def test_empty_output_and_no_files_is_no_evidence(tmp_path: Path) -> None:
    backend = _FakeBackend(SandboxExecutionResult(output="  \n", exit_code=0))

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is None
    assert capture.has_evidence is False


def test_artifact_files_alone_are_evidence(tmp_path: Path) -> None:
    artifact_root = tmp_path / "progress-artifacts" / "profiles" / "round-0001"

    def write_report() -> None:
        (artifact_root / "report.txt").write_text("hot: foo 40%\n")

    backend = _FakeBackend(SandboxExecutionResult(output="", exit_code=0), side_effect=write_report)

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is None
    assert capture.has_evidence is True
    assert capture.artifact_files == ("report.txt",)


@pytest.mark.parametrize(
    ("exit_code", "reason"),
    [(1, "exit code 1"), (TIMEOUT_EXIT_CODE, "timed out")],
)
def test_failed_command_warns_and_has_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, reason: str
) -> None:
    warnings = _capture_warnings(monkeypatch)
    stdout = json.dumps(_SUMMARY)
    backend = _FakeBackend(
        SandboxExecutionResult(output=stdout + "\nboom", exit_code=exit_code, stdout=stdout)
    )

    capture, _ = _run(tmp_path, backend)

    assert capture.summary is None, "a failed command's output is never trusted as structured"
    assert capture.has_evidence is False
    assert capture.exit_code == exit_code
    [(text, kw)] = warnings
    assert text == "custom profiler failed"
    assert reason in kw["detail"]
    assert "python profiler/profile.py" in kw["detail"]
    assert kw["round_label"] == "round-1-profiler"


# ---------------------------------------------------------------------------
# interpret_custom_profiler
# ---------------------------------------------------------------------------


def _capture(**overrides: object) -> CustomProfilerCapture:
    fields: dict[str, Any] = {
        "exit_code": 0,
        "output": "perf stat: 1,234 cycles",
        "artifact_location": "progress-artifacts/profiles/round-0001",
        "artifact_files": ("report.txt",),
        "summary": None,
    }
    fields.update(overrides)
    return CustomProfilerCapture(**fields)


_TURN = ProfilerTurn(system_prompt="SYSTEM", round_label="round-1-profiler")


def test_interpret_returns_the_structured_summary_without_an_agent_call() -> None:
    ctx = MagicMock()
    summary = ProfilerSummary.model_validate(_SUMMARY)

    result = interpret_custom_profiler(ctx, _capture(summary=summary), _TURN)

    assert result is summary
    ctx.invoke.assert_not_called()


def test_interpret_skips_the_agent_without_evidence() -> None:
    ctx = MagicMock()

    result = interpret_custom_profiler(ctx, _capture(exit_code=1), _TURN)

    assert result is None
    ctx.invoke.assert_not_called()


def test_interpret_invokes_the_agent_without_mcp_and_returns_its_summary() -> None:
    ctx = MagicMock()
    summary = ProfilerSummary.model_validate(_SUMMARY)
    ctx.invoke.return_value = summary

    result = interpret_custom_profiler(ctx, _capture(), _TURN)

    assert result is summary
    ctx.invoke.assert_called_once()
    kwargs = ctx.invoke.call_args.kwargs
    assert kwargs["kind"] == "profiler"
    assert kwargs["system_prompt"] == "SYSTEM"
    assert kwargs["round_label"] == "round-1-profiler"
    assert kwargs["mcp_servers"] is None
    assert kwargs["response_cls"] is ProfilerSummary
    assert "Interpret the captured profiler evidence" in kwargs["user_prompt"]
    fallback = kwargs["fallback_factory"]()
    assert fallback.suggestions == "Re-run profiling on the next round."


def test_interpret_uses_the_injected_invoke_wrapper() -> None:
    ctx = MagicMock()
    summary = ProfilerSummary.model_validate(_SUMMARY)
    seen: list[dict[str, object]] = []

    def invoke(**kwargs: object) -> ProfilerSummary:
        seen.append(kwargs)
        return summary

    result = interpret_custom_profiler(ctx, _capture(), _TURN, invoke=invoke)

    assert result is summary
    ctx.invoke.assert_not_called()
    assert len(seen) == 1
    assert seen[0]["mcp_servers"] is None


def test_interpret_turns_agent_failure_into_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _capture_warnings(monkeypatch)
    ctx = MagicMock()
    ctx.invoke.side_effect = RuntimeError("agent exploded")

    result = interpret_custom_profiler(ctx, _capture(), _TURN)

    assert result is None
    [(text, kw)] = warnings
    assert text == "profiler failed"
    assert kw["detail"] == "agent exploded"
    assert kw["round_label"] == "round-1-profiler"


def test_prompt_context_exposes_the_capture_for_custom_template() -> None:
    context = custom_profiler_prompt_context(_capture())

    assert context == {
        "captured_output": "perf stat: 1,234 cycles",
        "captured_output_limit": CUSTOM_PROFILER_OUTPUT_TAIL_CHARS,
        "artifact_files": ("report.txt",),
        "artifact_location": "progress-artifacts/profiles/round-0001",
    }


# ---------------------------------------------------------------------------
# custom_profiler_summary
# ---------------------------------------------------------------------------

_COMMAND = CustomProfilerCommand(command="python profiler/profile.py", timeout_seconds=30)


def _summary(
    tmp_path: Path,
    backend: _FakeBackend,
    *,
    prompts: list[CustomProfilerCapture],
    invoke: Callable[..., ProfilerSummary] | None = None,
) -> tuple[ProfilerSummary | None, LoopContext]:
    ctx = _ctx(tmp_path, backend)

    def system_prompt(capture: CustomProfilerCapture) -> str:
        prompts.append(capture)
        return "RENDERED"

    run = CustomProfilerRun(
        command=_COMMAND,
        artifact_root=tmp_path / "profiles" / "round-0001",
        round_label="round-1-profiler",
        fallback_suggestions="n/a",
    )
    result = custom_profiler_summary(ctx, run, system_prompt=system_prompt, invoke=invoke)
    return result, ctx


def test_summary_uses_structured_output_without_rendering_a_prompt(tmp_path: Path) -> None:
    stdout = json.dumps(_SUMMARY)
    backend = _FakeBackend(SandboxExecutionResult(output=stdout, exit_code=0, stdout=stdout))
    prompts: list[CustomProfilerCapture] = []

    result, ctx = _summary(tmp_path, backend, prompts=prompts)

    assert result == ProfilerSummary.model_validate(_SUMMARY)
    assert prompts == []
    cast("MagicMock", ctx).invoke.assert_not_called()
    [(command, timeout)] = backend.commands
    assert command.endswith(" python profiler/profile.py")
    assert timeout == 30


def test_summary_is_none_without_evidence_and_never_renders(tmp_path: Path) -> None:
    backend = _FakeBackend(SandboxExecutionResult(output="", exit_code=0))
    prompts: list[CustomProfilerCapture] = []
    seen: list[dict[str, object]] = []

    def invoke(**kwargs: object) -> ProfilerSummary:
        seen.append(kwargs)
        pytest.fail("no agent turn without evidence")

    result, ctx = _summary(tmp_path, backend, prompts=prompts, invoke=invoke)

    assert result is None
    assert prompts == []
    assert seen == []
    cast("MagicMock", ctx).invoke.assert_not_called()


def test_summary_renders_once_and_interprets_unstructured_output(tmp_path: Path) -> None:
    backend = _FakeBackend(SandboxExecutionResult(output="perf stat: 1,234 cycles", exit_code=0))
    prompts: list[CustomProfilerCapture] = []
    seen: list[dict[str, object]] = []
    summary = ProfilerSummary.model_validate(_SUMMARY)

    def invoke(**kwargs: object) -> ProfilerSummary:
        seen.append(kwargs)
        return summary

    result, ctx = _summary(tmp_path, backend, prompts=prompts, invoke=invoke)

    assert result is summary
    [capture] = prompts
    assert capture.output == "perf stat: 1,234 cycles"
    assert capture.summary is None
    [kwargs] = seen
    assert kwargs["system_prompt"] == "RENDERED"
    assert kwargs["round_label"] == "round-1-profiler"
    assert kwargs["mcp_servers"] is None
    fallback = cast("Callable[[], ProfilerSummary]", kwargs["fallback_factory"])()
    assert fallback.suggestions == "n/a"
    cast("MagicMock", ctx).invoke.assert_not_called()


# ---------------------------------------------------------------------------
# declared_custom_profiler
# ---------------------------------------------------------------------------

_MANIFEST = """version = 1
[agent]
domain = "generic"
[accuracy]
command = ["python", "acc.py"]
[benchmark]
command = ["python", "bench.py"]
"""


def _write_bundle(root: Path, *, profiler: str = "") -> Path:
    (root / "vibesys.input.toml").write_text(_MANIFEST + profiler)
    (root / "OBJECTIVE.md").write_text("objective\n")
    return root


def test_declared_custom_profiler_is_none_without_a_profiler_section(tmp_path: Path) -> None:
    bundle = load_input_bundle(_write_bundle(tmp_path))

    assert declared_custom_profiler(bundle) is None


def test_declared_custom_profiler_carries_the_display_command_and_timeout(
    tmp_path: Path,
) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "prof.sh").write_text("#!/bin/sh\n")
    bundle = load_input_bundle(
        _write_bundle(
            tmp_path,
            profiler='[profiler]\ncommand = ["bin/prof.sh", "a b"]\ntimeout_seconds = 9\n',
        )
    )

    assert declared_custom_profiler(bundle) == CustomProfilerCommand(
        command="bin/prof.sh 'a b'", timeout_seconds=9
    )


def test_declared_custom_profiler_timeout_defaults_to_none(tmp_path: Path) -> None:
    bundle = load_input_bundle(
        _write_bundle(tmp_path, profiler='[profiler]\ncommand = ["python", "prof.py"]\n')
    )

    assert declared_custom_profiler(bundle) == CustomProfilerCommand(
        command="python prof.py", timeout_seconds=None
    )
