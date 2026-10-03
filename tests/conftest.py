"""Shared pytest configuration.

CI runs four environments (two operating systems x two Python versions) and the
log of a failing run is only visible to whoever owns the repository. So a failure
has to name itself: GitHub turns ``::error`` lines into annotations that travel
with the run's check output and can be read without signing in. Locally this
prints nothing.

    python -m pytest tests            # the whole suite, failures summarised

A failure on one OS or one Python version is a bug; a failure on all of them is
usually a version-sensitive API, and the annotation is what makes that obvious.
"""
from __future__ import annotations

import os

#: GitHub renders the first lines of a step output as annotations; a long list
#: is truncated anyway, so cap it rather than flooding the run page.
MAX_ANNOTATIONS = 20


def _escape(text: str) -> str:
    """GitHub workflow commands break on a raw % or a newline."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for kind, level in (("failed", "error"), ("error", "error")):
        reports = terminalreporter.stats.get(kind, [])[:MAX_ANNOTATIONS]
        for report in reports:
            where = report.nodeid.replace("::", " ").replace("tests/", "", 1)
            last = getattr(report, "longrepr", None)
            reason = ""
            if isinstance(last, tuple):
                reason = str(last[-1]).strip().splitlines()[-1]
            elif last is not None:
                reason = str(last).strip().splitlines()[-1]
            terminalreporter.write_line(
                f"::{level} title={_escape(where)}::{_escape(reason[:200])}")