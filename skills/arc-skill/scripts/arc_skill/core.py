from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class ArcSkillError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / ".arc"

    @property
    def config(self) -> Path:
        return self.state / "config.json"

    @property
    def events(self) -> Path:
        return self.state / "events.jsonl"

    @property
    def activity(self) -> Path:
        return self.state / "activity.jsonl"

    @property
    def verification(self) -> Path:
        return self.state / "verify.json"

    @property
    def plan(self) -> Path:
        return self.state / "plan.json"

    @property
    def notes(self) -> Path:
        return self.state / "NOTES.md"

    @property
    def rules(self) -> Path:
        return self.root / "rules.py"

    @property
    def dossier(self) -> Path:
        return self.state / "dossier.json"

    @property
    def receipts(self) -> Path:
        return self.state / "receipts"

    @property
    def mutations(self) -> Path:
        return self.state / "mutations.jsonl"

    @property
    def broker(self) -> Path:
        return self.state / "broker.json"

    @property
    def socket(self) -> Path:
        digest = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:24]
        tail = Path(f"arc-skill-{os.getuid()}") / "sockets" / f"{digest}.sock"
        candidate = Path(tempfile.gettempdir()) / tail
        if len(str(candidate).encode()) > 90:
            # AF_UNIX paths are capped near 104 bytes; a long TMPDIR breaks bind().
            return Path("/tmp") / tail
        return candidate

    @property
    def lock(self) -> Path:
        return self.state / "run.lock"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_game_id(value: str) -> str:
    game_id = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9]{2,16}", game_id):
        raise ArcSkillError(f"invalid game id {value!r}")
    return game_id


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as error:
        raise ArcSkillError(f"corrupt JSON in {path}: {error}") from error


def append_jsonl(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {"timestamp": now_iso(), **dict(value)}
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    with path.open("a") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ArcSkillError(
                f"corrupt JSONL in {path} line {line_number}: {error}"
            ) from error
    return output


@contextlib.contextmanager
def run_lock(paths: RunPaths) -> Iterator[None]:
    paths.state.mkdir(parents=True, exist_ok=True)
    with paths.lock.open("a+") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


@contextlib.contextmanager
def command_status(paths: RunPaths, command: str) -> Iterator[None]:
    started_monotonic = time.monotonic()
    events = load_jsonl(paths.events)
    record = {
        "status": "RUNNING",
        "command": command,
        "event": events[-1]["id"] if events else None,
        "risk": "paid_live_action"
        if command in {"act", "commit", "reset"}
        else "offline",
        "started_at": now_iso(),
        "pid": os.getpid(),
    }
    append_jsonl(paths.activity, {"kind": "command_start", **record})
    try:
        yield
    except Exception as error:
        current = load_jsonl(paths.events)
        record.update(
            status="ERROR",
            finished_at=now_iso(),
            elapsed_seconds=time.monotonic() - started_monotonic,
            event=current[-1]["id"] if current else record["event"],
            error=str(error)[:500],
        )
        append_jsonl(paths.activity, {"kind": "command_end", **record})
        raise
    else:
        current = load_jsonl(paths.events)
        record.update(
            status="FINISHED",
            finished_at=now_iso(),
            elapsed_seconds=time.monotonic() - started_monotonic,
            event=current[-1]["id"] if current else record["event"],
        )
        append_jsonl(paths.activity, {"kind": "command_end", **record})


def require_run(paths: RunPaths) -> dict[str, Any]:
    config = read_json(paths.config)
    if not isinstance(config, dict):
        raise ArcSkillError(
            f"{paths.root} is not initialized; run `arc start GAME_ID`"
        )
    return config


def grid_to_rows(grid: Any) -> list[str]:
    array = np.asarray(grid, dtype=np.int16)
    if array.ndim != 2:
        raise ArcSkillError(f"frame must be 2-D, got {array.shape}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 15):
        raise ArcSkillError("ARC colors must be in 0..15")
    return ["".join(format(int(cell), "x") for cell in row) for row in array]


def rows_to_grid(rows: Sequence[str]) -> np.ndarray:
    if not rows:
        raise ArcSkillError("empty frame")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ArcSkillError("ragged frame encoding")
    try:
        return np.asarray(
            [[int(cell, 16) for cell in row] for row in rows], dtype=np.int16
        )
    except ValueError as error:
        raise ArcSkillError("invalid frame encoding") from error


def normalize_state(value: Any) -> str:
    return str(getattr(value, "value", value))


def normalize_available(values: Any) -> list[int]:
    output: set[int] = set()
    for value in values or ():
        raw = getattr(value, "value", value)
        if isinstance(raw, tuple):
            raw = raw[0]
        output.add(int(raw))
    return sorted(output)


def normalize_observation(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        raw_frames = response.get("frame", response.get("frames"))
        state = response.get("state", "NOT_FINISHED")
        completed = response.get("levels_completed", 0)
        total = response.get("win_levels", 1)
        available = response.get("available_actions", ())
    else:
        raw_frames = response.frame
        state = response.state
        completed = response.levels_completed
        total = response.win_levels
        available = response.available_actions
    frames = [np.asarray(frame, dtype=np.int16) for frame in raw_frames]
    if not frames:
        raise ArcSkillError("environment returned no frames")
    return {
        "state": normalize_state(state),
        "levels_completed": int(completed),
        "win_levels": int(total),
        "available_actions": normalize_available(available),
        "frames": [grid_to_rows(frame) for frame in frames],
    }


def make_event(
    response: Any,
    action: str,
    data: Mapping[str, int] | None,
    previous: Mapping[str, Any] | None,
    note: str = "",
) -> dict[str, Any]:
    observed = normalize_observation(response)
    return {
        "id": -1,
        "timestamp": now_iso(),
        "action": action,
        "data": dict(data) if data else None,
        "note": note,
        "state": observed["state"],
        "levels_completed": observed["levels_completed"],
        "level_before": None if previous is None else int(previous["levels_completed"]),
        "win_levels": observed["win_levels"],
        "available_actions": observed["available_actions"],
        "n_frames": len(observed["frames"]),
        "frames": observed["frames"],
        "counts_action": action != "START",
    }


def load_events(paths: RunPaths) -> list[dict[str, Any]]:
    events = load_jsonl(paths.events)
    for index, event in enumerate(events):
        if event.get("id") != index:
            raise ArcSkillError(
                f"event timeline is not contiguous at line {index + 1}"
            )
    return events


def append_event(paths: RunPaths, event: Mapping[str, Any]) -> dict[str, Any]:
    events = load_events(paths)
    record = {**dict(event), "id": len(events)}
    append_jsonl(paths.events, record)
    # append_jsonl injects a timestamp first, then record's timestamp wins.
    return record


def frame_at(event: Mapping[str, Any], frame: int = -1) -> np.ndarray:
    return rows_to_grid(event["frames"][frame])


def canonical_action(event: Mapping[str, Any]) -> str:
    data = event.get("data")
    if data and "x" in data and "y" in data:
        return f"{event['action']}:{data['x']},{data['y']}"
    return str(event["action"])


def parse_action(token: str) -> tuple[str, dict[str, int] | None]:
    name, separator, suffix = token.strip().upper().partition(":")
    if name == "RESET":
        if separator:
            raise ArcSkillError("RESET takes no coordinates")
        return name, None
    if (
        not name.startswith("ACTION")
        or not name[6:].isdigit()
        or not 1 <= int(name[6:]) <= 7
    ):
        raise ArcSkillError(
            f"invalid action {token!r}; use ACTION1..ACTION7 or ACTION6:x,y"
        )
    if name == "ACTION6":
        if not separator:
            raise ArcSkillError("ACTION6 requires coordinates: ACTION6:x,y")
        try:
            x_text, y_text = suffix.split(",", 1)
            x, y = int(x_text), int(y_text)
        except (ValueError, TypeError):
            raise ArcSkillError(f"invalid coordinate action {token!r}") from None
        if not (0 <= x <= 63 and 0 <= y <= 63):
            raise ArcSkillError("ACTION6 coordinates must be in 0..63")
        return name, {"x": x, "y": y}
    if separator:
        raise ArcSkillError(f"only ACTION6 accepts coordinates: {token!r}")
    return name, None


def action_number(token: str) -> int:
    name, _ = parse_action(token)
    if name == "RESET":
        return 0
    return int(name[6:])


def segment_start(events: Sequence[Mapping[str, Any]], index: int | None = None) -> int:
    if not events:
        return 0
    cursor = len(events) - 1 if index is None else index
    while cursor > 0:
        event = events[cursor]
        prior = events[cursor - 1]
        if (
            event["action"] == "RESET"
            or event["levels_completed"] != prior["levels_completed"]
        ):
            return cursor
        cursor -= 1
    return 0


def context_for(events: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    event = events[index]
    return {
        "event": index,
        "level": int(event["levels_completed"]) + 1,
        "levels_completed": int(event["levels_completed"]),
        "win_levels": int(event["win_levels"]),
        "environment_state": str(event["state"]),
        "available_actions": tuple(event["available_actions"]),
        "segment_start": segment_start(events, index),
    }


@contextlib.contextmanager
def import_path(path: Path, prefix: str) -> Iterator[ModuleType]:
    before_path = list(sys.path)
    before_modules = set(sys.modules)
    unique = f"_{prefix}_{time.time_ns()}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ArcSkillError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.path.insert(0, str(path.parent))
        sys.modules[unique] = module
        spec.loader.exec_module(module)
        yield module
    except ArcSkillError:
        raise
    except Exception as error:
        raise ArcSkillError(
            f"failed to load {path.name}: {type(error).__name__}: {error}"
        ) from error
    finally:
        sys.path[:] = before_path
        for name in set(sys.modules) - before_modules:
            loaded = sys.modules.get(name)
            source = getattr(loaded, "__file__", None)
            package_paths = [
                str(item) for item in (getattr(loaded, "__path__", ()) or ())
            ]
            if (
                name == unique
                or (source and str(source).startswith(str(path.parent)))
                or any(item.startswith(str(path.parent)) for item in package_paths)
            ):
                sys.modules.pop(name, None)


def check_public_action(token: str, available: Sequence[int]) -> None:
    number = action_number(token)
    if number not in {int(value) for value in available}:
        raise ArcSkillError(
            f"{token} is unavailable; public actions are {list(available)}"
        )
