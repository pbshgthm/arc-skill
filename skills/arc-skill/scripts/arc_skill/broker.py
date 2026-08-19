from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import (
    ArcSkillError,
    RunPaths,
    append_event,
    append_jsonl,
    atomic_json,
    load_events,
    load_jsonl,
    make_event,
    normalize_observation,
    read_json,
    rows_to_grid,
)


LOCAL_MODE = "local"
REMOTE_MODE = "competition"


def local_cache_root() -> Path:
    """Return the durable shared cache used by local public-game runs."""
    configured = os.getenv("ARC_SKILL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = os.getenv("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "arc-skill" / "arcade").resolve()


def _adopt_cached_game(environments: Path, game_id: str) -> None:
    """Copy an already-downloaded game from another local cache, if one exists."""
    if _cached_game_exists(environments, game_id):
        return
    xdg = os.getenv("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    donors = [Path(value).expanduser() for value in (os.getenv("AA3_CACHE_DIR"),) if value]
    donors.append(base / "aa3" / "arcade")
    for donor in donors:
        source = donor / "environment_files" / game_id.lower()
        if _cached_game_exists(donor / "environment_files", game_id):
            shutil.copytree(
                source, environments / game_id.lower(), dirs_exist_ok=True
            )
            return


def is_remote_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("mode", LOCAL_MODE)).lower() == REMOTE_MODE


def _cached_game_exists(environments: Path, game_id: str) -> bool:
    game = environments / game_id.lower()
    return game.is_dir() and any(game.glob("*/metadata.json"))


def _import_factory(spec: str, root: Path) -> Any:
    module_name, separator, attribute = spec.rpartition(":")
    if not separator:
        raise ArcSkillError("adapter must be module:factory or /path/file.py:factory")
    candidate = Path(module_name)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.suffix == ".py" and candidate.exists():
        info = importlib.util.spec_from_file_location(
            f"_arc_adapter_{time.time_ns()}", candidate
        )
        if info is None or info.loader is None:
            raise ArcSkillError(f"cannot import adapter {candidate}")
        module = importlib.util.module_from_spec(info)
        sys.modules[info.name] = module
        info.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ArcSkillError(f"adapter factory {spec!r} is not callable")
    return factory


class ArcSession:
    def __init__(self, root: Path, config: Mapping[str, Any]):
        try:
            import arc_agi
            from arc_agi import OperationMode
        except ImportError as error:
            raise ArcSkillError(
                "arc-agi is unavailable in the managed runtime"
            ) from error

        mode_name = str(config.get("mode", LOCAL_MODE)).lower()
        if mode_name not in {LOCAL_MODE, REMOTE_MODE}:
            raise ArcSkillError(f"unsupported arc-skill mode {mode_name!r}")
        self.remote = mode_name == REMOTE_MODE

        if self.remote:
            runtime = Path(tempfile.gettempdir()) / f"arc-skill-{os.getuid()}" / "remote"
            mode = OperationMode.COMPETITION
        else:
            runtime = Path(str(config.get("cache_dir") or local_cache_root()))
            _adopt_cached_game(
                runtime / "environment_files", str(config["game_id"])
            )
            game_cached = _cached_game_exists(
                runtime / "environment_files", str(config["game_id"])
            )
            mode = OperationMode.OFFLINE if game_cached else OperationMode.NORMAL
        environments = runtime / "environment_files"
        recordings = root / ".arc" / "recordings"
        environments.mkdir(parents=True, exist_ok=True)
        recordings.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("arc_skill.broker")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.arcade = arc_agi.Arcade(
            arc_api_key=os.getenv("ARC_API_KEY", ""),
            operation_mode=mode,
            environments_dir=str(environments),
            recordings_dir=str(recordings),
            logger=logger,
        )
        self.environment = self.arcade.make(
            str(config["game_id"]), seed=int(config.get("seed", 0))
        )
        if self.environment is None:
            if self.remote:
                raise ArcSkillError(
                    f"could not initialize remote competition run for {config['game_id']}; ARC_API_KEY and network access are required"
                )
            if not os.getenv("ARC_API_KEY"):
                raise ArcSkillError(
                    f"LOCAL_CACHE_MISS | {config['game_id']} is not in {environments}; set ARC_API_KEY once so arc-skill can download the public game into its durable local cache"
                )
            raise ArcSkillError(
                f"could not initialize local {config['game_id']}: the authenticated download or cached game failed"
            )

    @property
    def observation(self) -> Any:
        observed = self.environment.observation_space
        if observed is None:
            self._missing_observation()
        return observed

    @property
    def public_info(self) -> dict[str, Any]:
        info = getattr(self.environment, "info", None)
        if info is None:
            return {}
        output: dict[str, Any] = {}
        for name in ("game_id", "title", "tags", "default_fps"):
            value = getattr(info, name, None)
            if value is not None:
                output[name] = list(value) if isinstance(value, tuple) else value
        return output

    def step(
        self,
        action: str,
        data: dict[str, int] | None,
        reasoning: Mapping[str, Any] | None,
    ) -> Any:
        from arcengine import GameAction

        observed = self.environment.step(
            GameAction.from_name(action), data=data, reasoning=reasoning
        )
        if observed is None:
            self._missing_observation()
        return observed

    def _missing_observation(self) -> None:
        if self.remote:
            raise ArcSkillError(
                "REMOTE_SESSION_EXPIRED_OR_UNAVAILABLE | the competition server returned no observation. This remote run cannot be reconstructed; preserve its artifacts and use a fresh directory for another run"
            )
        raise ArcSkillError("the local simulator returned no observation")

    def finalize(self) -> None:
        self.arcade.close_scorecard()


def _create_session(root: Path, config: Mapping[str, Any]) -> Any:
    adapter = str(config.get("adapter", "arc_agi"))
    if adapter == "arc_agi":
        return ArcSession(root, config)
    return _import_factory(adapter, root)(root, config)


def _encode_observation(value: Any) -> dict[str, Any]:
    if value is None:
        raise ArcSkillError("environment returned no observation")
    return normalize_observation(value)


def _decode_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame": [rows_to_grid(rows) for rows in value["frames"]],
        "state": value["state"],
        "levels_completed": value["levels_completed"],
        "win_levels": value["win_levels"],
        "available_actions": value["available_actions"],
    }


def _request(
    paths: RunPaths, payload: Mapping[str, Any], timeout: float = 10.0
) -> dict[str, Any]:
    descriptor = read_json(paths.broker)
    config = read_json(paths.config, {})
    local = isinstance(config, dict) and not is_remote_config(config)
    recovery = (
        "run `arc start GAME_ID` to replay the journal and resume"
        if local
        else "a remote competition run cannot be reconstructed"
    )
    if not isinstance(descriptor, dict) or descriptor.get("status") != "READY":
        raise ArcSkillError(f"the environment owner is unavailable; {recovery}")
    token_path = paths.state / "broker.token"
    try:
        token = token_path.read_text().strip()
    except FileNotFoundError as error:
        raise ArcSkillError("environment owner token is missing") from error
    message = (
        json.dumps({"token": token, **dict(payload)}, separators=(",", ":")).encode()
        + b"\n"
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(paths.socket))
            client.sendall(message)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except (OSError, TimeoutError) as error:
        raise ArcSkillError(f"the environment owner stopped responding; {recovery}") from error
    if not chunks:
        raise ArcSkillError("empty response from environment owner")
    response = json.loads(b"".join(chunks).splitlines()[0])
    if not response.get("ok"):
        raise ArcSkillError(str(response.get("error", "environment owner error")))
    return response


def broker_ping(paths: RunPaths) -> bool:
    try:
        return bool(_request(paths, {"op": "ping"}, timeout=0.5).get("pong"))
    except ArcSkillError:
        return False


def broker_observe(paths: RunPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    response = _request(paths, {"op": "observe"})
    return _decode_observation(response["observation"]), dict(
        response.get("public_info") or {}
    )


def broker_step(
    paths: RunPaths,
    action: str,
    data: dict[str, int] | None,
    reasoning: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int, str | None]:
    response = _request(
        paths,
        {"op": "step", "action": action, "data": data, "reasoning": reasoning},
        timeout=30.0,
    )
    return (
        _decode_observation(response["observation"]),
        int(response["mutation_id"]),
        response.get("finalization_warning"),
    )


def broker_matches_latest_event(paths: RunPaths) -> bool:
    events = load_events(paths)
    if not events:
        return True
    observation, _ = broker_observe(paths)
    return _encode_observation(observation) == _event_observation(events[-1])


def reconcile_mutations(paths: RunPaths) -> int:
    """Recover a paid step journaled by the broker before a CLI process died."""
    events = load_events(paths)
    known = {
        int(event["mutation_id"])
        for event in events
        if event.get("mutation_id") is not None
    }
    recovered = 0
    for mutation in load_jsonl(paths.mutations):
        mutation_id = int(mutation["mutation_id"])
        if mutation_id in known:
            continue
        previous = events[-1] if events else None
        observation = _decode_observation(mutation["observation"])
        event = make_event(
            observation,
            mutation["action"],
            mutation.get("data"),
            previous,
            note="recovered from broker mutation journal",
        )
        event["mutation_id"] = mutation_id
        appended = append_event(paths, event)
        events.append(appended)
        known.add(mutation_id)
        recovered += 1
    if recovered:
        append_jsonl(
            paths.activity,
            {
                "kind": "mutation_recovery",
                "event": len(events) - 1,
                "recovered": recovered,
            },
        )
    return recovered


def start_broker(paths: RunPaths) -> None:
    paths.state.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    token_path = paths.state / "broker.token"
    token_path.write_text(token)
    token_path.chmod(0o600)
    descriptor = {
        "status": "STARTING",
        "started_at": time.time(),
        "mode": read_json(paths.config, {}).get("mode", LOCAL_MODE),
    }
    atomic_json(paths.broker, descriptor)
    log = (paths.state / "broker.log").open("ab", buffering=0)
    script = Path(__file__).resolve().parents[1] / "broker_server.py"
    process = subprocess.Popen(
        [sys.executable, str(script), "--run-dir", str(paths.root)],
        cwd=str(paths.root),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
    )
    current = read_json(paths.broker, {})
    if current.get("status") == "STARTING":
        atomic_json(paths.broker, {**current, "pid": process.pid})
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        current = read_json(paths.broker, {})
        if current.get("status") == "ERROR":
            raise ArcSkillError(
                str(current.get("error", "environment initialization failed"))
            )
        if current.get("status") == "READY" and broker_ping(paths):
            return
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise ArcSkillError("environment owner did not start; see .arc/broker.log")


def stop_broker(paths: RunPaths) -> None:
    descriptor = read_json(paths.broker, {})
    pid = descriptor.get("pid") if isinstance(descriptor, dict) else None
    if isinstance(pid, int):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _event_observation(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": str(event["state"]),
        "levels_completed": int(event["levels_completed"]),
        "win_levels": int(event["win_levels"]),
        "available_actions": [int(value) for value in event["available_actions"]],
        "frames": list(event["frames"]),
    }


def _competition_step(
    session: Any,
    action: str,
    data: dict[str, int] | None,
    reasoning: Mapping[str, Any] | None,
    *,
    fresh_level: bool,
) -> tuple[Any, bool]:
    """Apply one paid action while preventing a local opener reset from rewinding the game."""
    before = _encode_observation(session.observation)
    if action == "RESET" and fresh_level:
        observed = session.observation
    else:
        observed = session.step(action, data, reasoning)
    encoded = _encode_observation(observed)
    level_advanced = encoded["levels_completed"] != before["levels_completed"]
    return observed, action == "RESET" or level_advanced


def _replay_local_session(
    session: Any, paths: RunPaths
) -> tuple[list[dict[str, Any]], bool]:
    """Reconstruct an exact local session from the append-only paid-action journal."""
    events = load_events(paths)
    current = _encode_observation(session.observation)
    if events and current != _event_observation(events[0]):
        raise ArcSkillError(
            "LOCAL_REPLAY_DIVERGED | fresh simulator state differs from event 0; cached game or seed changed"
        )
    mutations = load_jsonl(paths.mutations)
    fresh_level = True
    for mutation in mutations:
        observed, fresh_level = _competition_step(
            session,
            str(mutation["action"]),
            mutation.get("data"),
            mutation.get("reasoning"),
            fresh_level=fresh_level,
        )
        actual = _encode_observation(observed)
        expected = dict(mutation["observation"])
        if actual != expected:
            raise ArcSkillError(
                "LOCAL_REPLAY_DIVERGED | mutation "
                f"{mutation.get('mutation_id')} no longer reproduces its recorded observation"
            )
    return mutations, fresh_level


def serve(paths: RunPaths) -> None:
    config = read_json(paths.config)
    if not isinstance(config, dict):
        return
    try:
        session = _create_session(paths.root, config)
        if is_remote_config(config):
            mutations = load_jsonl(paths.mutations)
            fresh_level = False
        else:
            mutations, fresh_level = _replay_local_session(session, paths)
        token = (paths.state / "broker.token").read_text().strip()
        try:
            paths.socket.unlink()
        except FileNotFoundError:
            pass
        paths.socket.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(paths.socket))
        paths.socket.chmod(0o600)
        server.listen(4)
        atomic_json(
            paths.broker,
            {
                "status": "READY",
                "pid": os.getpid(),
                "mode": config.get("mode", LOCAL_MODE),
                "replayed_mutations": len(mutations),
                "started_at": time.time(),
            },
        )
    except Exception as error:  # noqa: BLE001 - broker persists arbitrary adapter failures
        atomic_json(
            paths.broker,
            {
                "status": "ERROR",
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        return

    sequence = max((int(item.get("mutation_id", 0)) for item in mutations), default=0)
    while True:
        connection, _ = server.accept()
        terminal = False
        with connection:
            try:
                raw = b""
                while b"\n" not in raw:
                    chunk = connection.recv(1 << 20)
                    if not chunk:
                        break
                    raw += chunk
                request = json.loads(raw.splitlines()[0])
                if not secrets.compare_digest(str(request.get("token", "")), token):
                    raise ArcSkillError("invalid environment owner token")
                operation = request.get("op")
                if operation == "ping":
                    response = {"ok": True, "pong": True}
                elif operation == "observe":
                    response = {
                        "ok": True,
                        "observation": _encode_observation(session.observation),
                        "public_info": getattr(session, "public_info", {}),
                    }
                elif operation == "step":
                    if is_remote_config(config):
                        observed = session.step(
                            request["action"],
                            request.get("data"),
                            request.get("reasoning"),
                        )
                        if observed is None:
                            raise ArcSkillError(
                                "REMOTE_SESSION_EXPIRED_OR_UNAVAILABLE | the competition server returned no observation. This remote run cannot be reconstructed; preserve its artifacts and use a fresh directory for another run"
                            )
                    else:
                        observed, fresh_level = _competition_step(
                            session,
                            request["action"],
                            request.get("data"),
                            request.get("reasoning"),
                            fresh_level=fresh_level,
                        )
                    encoded = _encode_observation(observed)
                    sequence += 1
                    append_jsonl(
                        paths.mutations,
                        {
                            "mutation_id": sequence,
                            "action": request["action"],
                            "data": request.get("data"),
                            "reasoning": request.get("reasoning"),
                            "observation": encoded,
                        },
                    )
                    response = {
                        "ok": True,
                        "observation": encoded,
                        "mutation_id": sequence,
                    }
                    terminal = encoded["state"] == "WIN"
                    if terminal and callable(getattr(session, "finalize", None)):
                        try:
                            session.finalize()
                        except Exception as error:  # noqa: BLE001 - action is already journaled
                            response["finalization_warning"] = (
                                f"{type(error).__name__}: {error}"
                            )
                else:
                    raise ArcSkillError(f"unknown broker operation {operation!r}")
            except Exception as error:  # noqa: BLE001 - isolate arbitrary adapter failures
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            connection.sendall(
                json.dumps(response, separators=(",", ":")).encode() + b"\n"
            )
        if terminal:
            atomic_json(
                paths.broker,
                {
                    "status": "FINISHED",
                    "pid": os.getpid(),
                    "mode": config.get("mode", LOCAL_MODE),
                    "finished_at": time.time(),
                },
            )
            server.close()
            paths.socket.unlink(missing_ok=True)
            return
