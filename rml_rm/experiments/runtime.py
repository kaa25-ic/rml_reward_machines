"""Runtime helpers shared by experiment scripts."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from rml_rm.monitors import RMLMonitorProcess, find_free_port


@dataclass(frozen=True)
class MonitorPairRuntime:
    """Runtime files and ports for train/eval monitor processes."""

    train_port: int
    eval_port: int
    train_config_path: Path
    eval_config_path: Path


@dataclass(frozen=True)
class MonitorRuntime:
    """Runtime files and port for a single monitor process."""

    port: int
    config_path: Path


@dataclass(frozen=True)
class MonitorGroupRuntime:
    """Runtime files and ports for a group of monitor processes."""

    ports: dict[str, int]
    config_paths: dict[str, Path]


def configure_global_seed(seed: int | None) -> None:
    """Configure Python and NumPy random seeds."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def configure_torch_seed(seed: int | None) -> None:
    """Configure Python, NumPy, and PyTorch random seeds."""
    configure_global_seed(seed)
    if seed is None:
        return
    import torch

    torch.manual_seed(seed)


def resolve_torch_device(device_name: str):
    """Resolve an auto/cpu/cuda/mps device string for PyTorch code."""
    import torch

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON after converting common non-JSON values."""
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, sort_keys: bool = False) -> None:
    """Write JSON lines after converting common non-JSON values."""
    path.write_text(
        "".join(json.dumps(json_ready(row), sort_keys=sort_keys) + "\n" for row in rows),
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_runtime_monitor_config(
    path: Path,
    *,
    template_path: Path,
    port: int,
    max_episode_steps: int | None = None,
) -> Path:
    """Write a monitor YAML config for a runtime port."""
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["host"] = "127.0.0.1"
    config["port"] = int(port)
    if max_episode_steps is not None:
        config["max_episode_steps"] = int(max_episode_steps)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def allocate_monitor_ports() -> tuple[int, int]:
    """Allocate distinct train/eval monitor ports."""
    train_port = find_free_port()
    eval_port = find_free_port()
    while eval_port == train_port:
        eval_port = find_free_port()
    return train_port, eval_port


def allocate_monitor_port_group(keys: list[str] | tuple[str, ...]) -> dict[str, int]:
    """Allocate one distinct monitor port per key."""
    ports: dict[str, int] = {}
    used: set[int] = set()
    for key in keys:
        port = find_free_port()
        while port in used:
            port = find_free_port()
        ports[str(key)] = port
        used.add(port)
    return ports


@contextmanager
def managed_monitor_pair(
    *,
    output_dir: Path,
    monitor_config_template: Path,
    monitor_spec_path: Path,
    max_episode_steps: int | None = None,
) -> Iterator[MonitorPairRuntime]:
    """Start paired train/eval RML monitors and stop them on exit."""
    train_port, eval_port = allocate_monitor_ports()
    runtime = MonitorPairRuntime(
        train_port=train_port,
        eval_port=eval_port,
        train_config_path=write_runtime_monitor_config(
            output_dir / "monitor_train_config.yaml",
            template_path=monitor_config_template,
            port=train_port,
            max_episode_steps=max_episode_steps,
        ),
        eval_config_path=write_runtime_monitor_config(
            output_dir / "monitor_eval_config.yaml",
            template_path=monitor_config_template,
            port=eval_port,
            max_episode_steps=max_episode_steps,
        ),
    )
    train_monitor = RMLMonitorProcess(
        spec_path=monitor_spec_path,
        port=train_port,
        log_path=output_dir / "train_rml_monitor.log",
    )
    eval_monitor = RMLMonitorProcess(
        spec_path=monitor_spec_path,
        port=eval_port,
        log_path=output_dir / "eval_rml_monitor.log",
    )
    try:
        train_monitor.start()
        eval_monitor.start()
        yield runtime
    finally:
        train_monitor.stop()
        eval_monitor.stop()


@contextmanager
def managed_monitor_group(
    *,
    output_dir: Path,
    monitor_specs: dict[str, Path],
    monitor_config_templates: dict[str, Path],
    config_dir_name: str,
    log_dir_name: str,
    max_episode_steps: int | None = None,
) -> Iterator[MonitorGroupRuntime]:
    """Start a keyed group of RML monitors and stop them on exit."""
    keys = tuple(sorted(monitor_specs))
    if set(keys) != set(monitor_config_templates):
        raise ValueError("Monitor specs and config templates must use the same keys.")

    ports = allocate_monitor_port_group(keys)
    config_dir = output_dir / config_dir_name
    log_dir = output_dir / log_dir_name
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_paths: dict[str, Path] = {}
    monitors: list[RMLMonitorProcess] = []
    for key in keys:
        config_paths[key] = write_runtime_monitor_config(
            config_dir / f"{key}.yaml",
            template_path=monitor_config_templates[key],
            port=ports[key],
            max_episode_steps=max_episode_steps,
        )
        monitors.append(
            RMLMonitorProcess(
                spec_path=monitor_specs[key],
                port=ports[key],
                log_path=log_dir / f"{key}.log",
            )
        )

    try:
        for monitor in monitors:
            monitor.start()
        yield MonitorGroupRuntime(ports=ports, config_paths=config_paths)
    finally:
        for monitor in reversed(monitors):
            monitor.stop()


@contextmanager
def managed_monitor(
    *,
    output_dir: Path,
    monitor_config_template: Path,
    monitor_spec_path: Path,
    log_name: str = "rml_monitor.log",
    config_name: str = "monitor_config.yaml",
    max_episode_steps: int | None = None,
) -> Iterator[MonitorRuntime]:
    """Start one RML monitor and stop it on exit."""
    port = find_free_port()
    runtime = MonitorRuntime(
        port=port,
        config_path=write_runtime_monitor_config(
            output_dir / config_name,
            template_path=monitor_config_template,
            port=port,
            max_episode_steps=max_episode_steps,
        ),
    )
    monitor = RMLMonitorProcess(
        spec_path=monitor_spec_path,
        port=port,
        log_path=output_dir / log_name,
    )
    try:
        monitor.start()
        yield runtime
    finally:
        monitor.stop()


def read_monitor_csv(path: Path) -> pd.DataFrame:
    """Read an SB3 Monitor CSV with normalized column names."""
    if not path.exists():
        return pd.DataFrame(
            columns=["episode_return", "episode_length", "elapsed_time_seconds"]
        )
    frame = pd.read_csv(path, skiprows=1)
    return frame.rename(
        columns={
            "r": "episode_return",
            "l": "episode_length",
            "t": "elapsed_time_seconds",
        }
    )


def rename_monitor_csv_columns(path: Path) -> None:
    """Rename SB3 Monitor CSV columns in place."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[1] != "r,l,t":
        return
    lines[1] = "episode_return,episode_length,elapsed_time_seconds"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
