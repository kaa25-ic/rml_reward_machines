"""Opt-in integration checks for the real Prolog RML monitor process."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from rml_rm.monitors.process import RMLMonitorProcess, find_free_port


RUN_INTEGRATION = os.environ.get("RUN_RML_INTEGRATION") == "1"


def _has_swipl() -> bool:
    if shutil.which("swipl") is not None:
        return True
    bundled_candidates = [
        Path("legacy/SWI-Prolog.app/Contents/MacOS/swipl"),
        Path("SWI-Prolog.app/Contents/MacOS/swipl"),
    ]
    return any(path.exists() and os.access(path, os.X_OK) for path in bundled_candidates)


@pytest.mark.integration
@pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_RML_INTEGRATION=1 to run")
@pytest.mark.skipif(not _has_swipl(), reason="SWI-Prolog executable not available")
def test_real_rml_monitor_process_starts_and_stops(tmp_path) -> None:
    monitor = RMLMonitorProcess(
        spec_path=Path("../../../envs/lunar_lander/specs/lunar_lander_protocol.pl"),
        port=find_free_port(),
        log_path=tmp_path / "monitor.log",
    )

    monitor.start()
    try:
        assert monitor.resolved_log_path is not None
        assert monitor.resolved_log_path.exists()
    finally:
        monitor.stop()
