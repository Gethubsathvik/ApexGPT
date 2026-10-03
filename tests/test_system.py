"""Live system-scan tests: every probe degrades, none of them raise."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core import system
from apexgpt.core.system import (ProcessSnapshot, SystemLoad, available_ram_gb,
                                cpu_percent, disk_usage_gb, load_average,
                                process_count, snapshot, top_processes)


# ------------------------------------------------------------------- probes
def test_cpu_percent_is_a_percentage_or_none():
    value = cpu_percent(sample=0.05)
    assert value is None or 0.0 <= value <= 100.0


def test_a_saturated_core_reports_high_utilisation():
    """Sampled from a subprocess: a GIL-bound thread cannot be measured.

    Deliberately loose - whether a given sample lands at 60% or 90% depends on
    the machine - but a fully busy core can never read as idle.
    """
    import subprocess
    import sys as _sys
    busy = subprocess.Popen(
        [_sys.executable, "-c",
         "x=0\nwhile True:\n    x = (x * 31 + 7) % 1000003"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        value = cpu_percent(sample=0.5)
    finally:
        busy.kill()
        busy.wait(timeout=5)
    if value is None:
        pytest.skip("CPU utilisation is not measurable on this platform")
    assert value > 5.0


def test_proc_stat_arithmetic():
    """idle + iowait is idle; the rest is work."""
    idle, total = system._split_cpu_line(["100.0", "0.0", "50.0", "40.0", "10.0"])
    assert idle == 50.0 and total == 200.0


def test_available_ram_is_positive_and_below_total():
    from apexgpt.core.device import total_ram_gb
    available = available_ram_gb()
    assert available is not None
    assert 0 < available <= total_ram_gb() + 0.1


def test_load_average_is_none_or_three_numbers():
    value = load_average()
    assert value is None or (len(value) == 3 and all(v >= 0 for v in value))


def test_disk_usage_adds_up(tmp_path):
    total, free, percent = disk_usage_gb(tmp_path)
    assert total > 0
    assert 0 <= free <= total
    assert 0 <= percent <= 100


def test_disk_usage_walks_up_to_an_existing_parent(tmp_path):
    missing = tmp_path / "a" / "b" / "c"
    total, free, _ = disk_usage_gb(missing)
    assert total > 0 and free > 0


def test_process_count_is_a_positive_int_or_none():
    count = process_count()
    assert count is None or count >= 1


def test_top_processes_are_optional_but_sorted(tmp_path):
    busy, hogs = top_processes(limit=3)
    if not system.has_psutil():
        assert busy == [] and hogs == []
        return
    for proc in busy:
        assert isinstance(proc, ProcessSnapshot) and proc.pid > 0
    cpus = [p.cpu_percent for p in busy if p.cpu_percent]
    assert cpus == sorted(cpus, reverse=True)
    assert len(busy) <= 3


# ---------------------------------------------------------------- snapshot
def test_snapshot_reports_a_complete_picture():
    load = snapshot()
    assert load.ram_total_gb > 0
    assert load.ram_available_gb > 0
    assert load.disk_total_gb > 0
    assert 0 <= load.ram_percent <= 100
    assert load.disk_path


def test_snapshot_text_mentions_the_headline_numbers():
    lines = "\n".join(snapshot().to_text())
    for label in ("CPU in use", "RAM", "disk", "processes"):
        assert label in lines


def test_snapshot_never_raises_for_a_bogus_path():
    load = snapshot(path=Path("/definitely/not/here"))
    assert load.ram_total_gb > 0


def test_snapshot_degrades_without_psutil(monkeypatch):
    monkeypatch.setattr(system, "_psutil", lambda: None)
    load = snapshot(sample=0.05)
    assert load.has_psutil is False
    assert load.busy_processes == []
    assert any("psutil" in note for note in load.notes)


# --------------------------------------------------------------- decisions
def test_busy_and_memory_pressure_thresholds():
    idle = SystemLoad(cpu_percent=10, ram_available_gb=6.0)
    assert idle.busy is False and idle.memory_pressure is False

    hot = SystemLoad(cpu_percent=95, ram_available_gb=6.0)
    assert hot.busy is True

    tight = SystemLoad(cpu_percent=10, ram_available_gb=0.4)
    assert tight.memory_pressure is True


def test_unknown_metrics_are_treated_as_idle():
    unknown = SystemLoad()
    assert unknown.busy is False
    assert unknown.memory_pressure is False
    assert "unknown" in "\n".join(unknown.to_text())


def test_to_dict_is_json_friendly():
    import json
    payload = json.loads(json.dumps(snapshot(sample=0.05).to_dict()))
    assert payload["busy"] in (True, False)
    assert "ram_available_gb" in payload
    assert isinstance(payload["busy_processes"], list)