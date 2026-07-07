"""Process manager for the external Prolog RML monitor."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from typing import TextIO


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def vendored_rml_root() -> Path:
    """Return the vendored Prolog monitor implementation directory."""
    return Path(__file__).resolve().parent / "rml"


@dataclass
class RMLMonitorProcess:
    """Start and stop one RML WebSocket monitor process."""

    spec_path: str | Path
    port: int
    rml_dir: str | Path | None = None
    monitor_script: str = "online_monitor_edit.sh"
    host: str = "127.0.0.1"
    startup_timeout_seconds: float = 10.0
    poll_interval_seconds: float = 0.1
    log_path: str | Path | None = None

    def __post_init__(self) -> None:
        self.rml_dir = Path(self.rml_dir) if self.rml_dir is not None else vendored_rml_root()
        self.spec_path = Path(self.spec_path)
        self._process: subprocess.Popen | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._log_file: TextIO | None = None
        self.resolved_log_path: Path | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> "RMLMonitorProcess":
        """Start the monitor and wait until its TCP port accepts connections."""
        if self._process is not None:
            raise RuntimeError("RML monitor process is already running.")

        script_path = Path(self.rml_dir) / self.monitor_script
        spec_argument = self._spec_argument()
        if not script_path.exists():
            raise FileNotFoundError(f"Monitor script not found: {script_path}")
        if not self._resolved_spec_path().exists():
            raise FileNotFoundError(f"Monitor spec not found: {self._resolved_spec_path()}")

        self._open_log()
        self._process = subprocess.Popen(
            [str(script_path), spec_argument, str(self.port)],
            cwd=str(self.rml_dir),
            stdin=subprocess.PIPE,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        """Terminate the monitor process and close log resources."""
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            self._terminate_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process, signal.SIGKILL)
                process.wait(timeout=2)

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> "RMLMonitorProcess":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    self._startup_error("RML monitor exited before accepting connections.")
                )
            if self._port_accepts_connections():
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(self._startup_error("Timed out waiting for RML monitor to start."))

    def _port_accepts_connections(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.25):
                return True
        except OSError:
            return False

    def _open_log(self) -> None:
        if self.log_path is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="rml-monitor-")
            self.resolved_log_path = Path(self._temp_dir.name) / "monitor.log"
        else:
            self.resolved_log_path = Path(self.log_path)
            self.resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.resolved_log_path.open("w", encoding="utf-8")

    def _spec_argument(self) -> str:
        resolved = self._resolved_spec_path()
        try:
            return str(resolved.relative_to(Path(self.rml_dir)))
        except ValueError:
            return str(resolved)

    def _resolved_spec_path(self) -> Path:
        if self.spec_path.is_absolute():
            return self.spec_path
        return Path(self.rml_dir) / self.spec_path

    def _startup_error(self, message: str) -> str:
        log_excerpt = ""
        if self.resolved_log_path is not None and self.resolved_log_path.exists():
            content = self.resolved_log_path.read_text(encoding="utf-8", errors="replace")
            log_excerpt = content[-2000:]
        if log_excerpt:
            return f"{message}\nMonitor log ({self.resolved_log_path}):\n{log_excerpt}"
        return message
