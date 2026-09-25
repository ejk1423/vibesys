"""Resuming a run that selected its bundle-declared ``[profiler]`` command."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from tests.entrypoints.test_headless import (
    _agent_configuration,
    _write_input_project,
    _write_project_run,
)

from entrypoints.cli import parse_cli_invocation
from vibesys.errors import ConfigurationError
from vibesys.profilers import ProfilerKind

if TYPE_CHECKING:
    from pathlib import Path

    from vs_project.api import AgentRunConfiguration

_RUN_ID = "20260811-120000-11111111-agent"
_RECORDED_COMMAND = "python prof.py --fast"


def _custom_profiler_configuration() -> AgentRunConfiguration:
    return _agent_configuration().model_copy(
        update={"profiler": "none", "profiler_command": _RECORDED_COMMAND}
    )


def _write_custom_profiler_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    declared: str | None = '["python", "prof.py", "--fast"]',
    configuration: AgentRunConfiguration | None = None,
) -> Path:
    project = _write_input_project(tmp_path)
    if declared is not None:
        manifest = project / "vibesys.input.toml"
        manifest.write_text(manifest.read_text() + f"\n[profiler]\ncommand = {declared}\n")
    _write_project_run(
        project,
        _RUN_ID,
        configuration=configuration or _custom_profiler_configuration(),
        created_at=datetime(2026, 8, 11, 12, tzinfo=UTC),
    )
    monkeypatch.chdir(project)
    return project


def test_resume_restores_auto_so_the_recorded_command_is_reselected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_custom_profiler_run(tmp_path, monkeypatch)

    args = parse_cli_invocation(["--resume", _RUN_ID]).args

    assert args.profiler is ProfilerKind.AUTO
    assert args.input_bundle.profiler_command_display == _RECORDED_COMMAND


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("auto", ProfilerKind.AUTO), ("none", ProfilerKind.NONE)],
)
def test_resume_accepts_an_explicit_auto_or_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    expected: ProfilerKind,
) -> None:
    _write_custom_profiler_run(tmp_path, monkeypatch)

    args = parse_cli_invocation(["--resume", _RUN_ID, "--profiler", flag]).args

    assert args.profiler is expected


def test_resume_rejects_a_built_in_profiler_over_the_recorded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_custom_profiler_run(tmp_path, monkeypatch)

    with pytest.raises(ConfigurationError) as exc:
        parse_cli_invocation(["--resume", _RUN_ID, "--profiler", "linux_cpu"])

    assert exc.value.diagnostic.code == "project_resume_configuration_mismatch"
    assert "profiler" in exc.value.diagnostic.message


@pytest.mark.parametrize(
    "declared",
    ['["python", "prof.py", "--slow"]', None],
    ids=["changed", "removed"],
)
def test_resume_rejects_a_bundle_that_no_longer_declares_the_recorded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: str | None
) -> None:
    _write_custom_profiler_run(tmp_path, monkeypatch, declared=declared)

    with pytest.raises(ConfigurationError) as exc:
        parse_cli_invocation(["--resume", _RUN_ID])

    assert exc.value.diagnostic.code == "project_resume_configuration_mismatch"
    assert "profiler.command" in exc.value.diagnostic.message


def test_resume_without_a_recorded_command_restores_the_recorded_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that recorded ``none`` without a command stays ``none`` on resume."""
    _write_custom_profiler_run(tmp_path, monkeypatch, configuration=_agent_configuration())

    args = parse_cli_invocation(["--resume", _RUN_ID]).args

    assert args.profiler is ProfilerKind.NONE
