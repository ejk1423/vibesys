"""Tests for ``vibesys.loops.roles`` -- the advertised loop-to-agent-roles contract.

Rather than asserting on the registry's literal contents (which would just
restate ``roles.py`` and pass unchanged if a call site drifted), these scan
each loop package's own source for the ``kind="<role>"`` literals its
``LoopContext.invoke`` call sites actually pass, and check that the
registry names exactly that set. A call site added or removed without a
matching registry update fails this test.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING

import vibesys.loops.agent as agent_package
import vibesys.loops.evolve as evolve_package
import vibesys.loops.plain as plain_package
from vibesys.loops.roles import EXPECTED_AGENT_ROLES

if TYPE_CHECKING:
    from types import ModuleType

# Matches a keyword argument like ``kind="profiler"`` (no surrounding spaces,
# per this repo's ruff-formatted style for keyword arguments) but not a plain
# assignment such as ``kind = "migrant"``, which ruff formats with spaces.
_KIND_KWARG = re.compile(r'\bkind="(\w+)"')

# The ``vibesys.loops.profiler`` entry points wrap ``ctx.invoke(kind="profiler", ...)``
# without spelling the literal out at their call sites, so a call to any of
# them implies the "profiler" role.
_INVOKE_PROFILER_CALL = re.compile(
    r"\b(?:invoke_profiler|invoke_profiler_agent|custom_profiler_summary|interpret_custom_profiler)\("
)

_LOOP_PACKAGES: dict[str, ModuleType] = {
    "agent": agent_package,
    "plain": plain_package,
    "evolve": evolve_package,
}


def _invoked_roles(package: ModuleType) -> set[str]:
    """Every agent role ``package``'s source actually invokes."""
    package_dir = Path(inspect.getfile(package)).parent
    roles: set[str] = set()
    for source_path in package_dir.rglob("*.py"):
        text = source_path.read_text()
        roles.update(_KIND_KWARG.findall(text))
        if _INVOKE_PROFILER_CALL.search(text):
            roles.add("profiler")
    return roles


def test_registry_matches_the_invoke_sites_for_every_loop() -> None:
    """``EXPECTED_AGENT_ROLES`` must name exactly the roles each loop can invoke."""
    for loop_kind, package in _LOOP_PACKAGES.items():
        assert _invoked_roles(package) == set(EXPECTED_AGENT_ROLES[loop_kind]), loop_kind
