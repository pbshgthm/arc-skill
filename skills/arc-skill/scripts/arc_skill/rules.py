"""Optional executable-rules tier: a plain rules.py, replay-verified, searched.

The agent writes rules.py in the run directory when a level is worth modeling.
`arc rules replay` checks it against every recorded transition; `arc rules
solve` searches it for a plan and saves per-step predictions that `arc commit`
verifies live, halting on the first surprise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import Unknown
from .core import (
    ArcSkillError,
    RunPaths,
    atomic_json,
    canonical_action,
    context_for,
    frame_at,
    grid_to_rows,
    import_path,
    load_events,
    now_iso,
)
from .evidence import observation_hash

RULES_HELP = """\
RULES CONTRACT | plain python in ./rules.py | action tokens like "ACTION1", "ACTION6:12,5"
required functions:
  initial(grid, context) -> state   ground a state from ANY settled board (numpy int grid).
                                    context: level, levels_completed, available_actions,
                                    environment_state, event, segment_start.
                                    Return Unknown("why") when a board cannot be grounded yet.
  step(state, action) -> state      the modeled result. Return the same state for a no-op,
                                    Unknown("why") for unmodeled, None for impossible-here.
  actions(state) -> iterable        tokens worth trying from this state (include "ACTION6:x,y").
  goal(state) -> bool               True exactly when this state completes the current level.
  observe(state) -> value           small JSON-able summary; checked against reality.
optional functions:
  key(state) -> hashable            dedupe key for search (default: the whole state).
  heuristic(state) -> float         admissible lower bound on remaining actions (default 0).
  render(state) -> grid             exact predicted board; upgrades checks to pixel-perfect.
  dead(state) -> bool | str         truthy when the state is lost; pruned during search.
Keep State a frozen dataclass. Model only verified mechanics; mark gaps Unknown — replay
reports them as INCOMPLETE, never as a fit. Import Unknown with:
  from arc_skill import Unknown
verify:  arc rules replay   (every recorded transition must fit or be an acknowledged gap)
search:  arc rules solve    (writes .arc/plan.json; execute with `arc commit @.arc/plan.json`)
"""

RULES_TEMPLATE = '''"""Executable rules for this game. Model only verified mechanics."""

from dataclasses import dataclass

import numpy as np

from arc_skill import Unknown


@dataclass(frozen=True)
class State:
    placeholder: int = 0


def initial(grid, context):
    return Unknown("grounding not implemented yet")


def step(state, action):
    return Unknown("no mechanics modeled yet")


def actions(state):
    return ()


def goal(state):
    return False


def observe(state):
    return {"placeholder": state.placeholder}
'''


def rules_hash(paths: RunPaths) -> str | None:
    try:
        return hashlib.sha256(paths.rules.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def init_rules(paths: RunPaths) -> Path:
    if paths.rules.exists():
        raise ArcSkillError(f"{paths.rules} already exists; edit it directly")
    paths.rules.write_text(RULES_TEMPLATE)
    return paths.rules


def _is_unknown(value: Any) -> bool:
    return isinstance(value, Unknown) or type(value).__name__ == "Unknown"


def _unknown_reason(value: Any) -> str:
    return str(getattr(value, "reason", "unspecified"))


def _freeze(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _freeze(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_freeze(item) for item in value), key=repr)
    if isinstance(value, Sequence):
        return [_freeze(item) for item in value]
    return repr(value)


def _call(module: Any, name: str, *arguments: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise ArcSkillError(f"rules.py is missing required function {name}()")
    try:
        return function(*arguments)
    except ArcSkillError:
        raise
    except Exception as error:
        raise ArcSkillError(
            f"rules.py {name}() failed: {type(error).__name__}: {error}"
        ) from error


def _optional(module: Any, name: str) -> Any:
    function = getattr(module, name, None)
    return function if callable(function) else None


def check_contract(module: Any) -> None:
    missing = [
        name
        for name in ("initial", "step", "actions", "goal", "observe")
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise ArcSkillError(
            f"rules.py is missing required functions: {', '.join(missing)}\n{RULES_HELP}"
        )


def _ground(module: Any, grid: np.ndarray, context: Mapping[str, Any]) -> tuple[Any, str | None]:
    value = _call(module, "initial", np.asarray(grid), dict(context))
    if value is None:
        raise ArcSkillError(
            "rules.py initial() returned None; return a state or Unknown('why')"
        )
    if _is_unknown(value):
        return None, _unknown_reason(value)
    return value, None


def _walk(
    module: Any, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replay every recorded transition through rules.py, re-grounding at gaps."""
    render = _optional(module, "render")
    dead = _optional(module, "dead")
    state: Any = None
    transitions = explained = 0
    pixel_checks = pixel_matches = 0
    gaps: list[str] = []
    first_mismatch: dict[str, Any] | None = None

    def gap(reason: str) -> None:
        if len(gaps) < 8 or reason not in gaps:
            gaps.append(reason)

    for index, event in enumerate(events):
        grid = frame_at(event)
        context = context_for(events, index)
        if index == 0 or event["action"] == "RESET":
            state, reason = _ground(module, grid, context)
            if reason is not None:
                gap(f"e{index}: cannot ground opener ({reason})")
            continue
        prior = events[index - 1]
        token = canonical_action(event)
        if state is None:
            state, reason = _ground(module, grid, context)
            if reason is not None:
                gap(f"e{index}: state lost and board not groundable ({reason})")
            continue
        transitions += 1
        predicted = _call(module, "step", state, token)
        advanced = int(event["levels_completed"]) > int(prior["levels_completed"])
        if advanced:
            if _is_unknown(predicted):
                gap(f"e{index} {token}: level-completing move unmodeled ({_unknown_reason(predicted)})")
            elif predicted is None:
                first_mismatch = {
                    "event": index,
                    "action": token,
                    "detail": "step() returned None (impossible) but this action completed the level",
                }
                break
            elif not _call(module, "goal", predicted):
                first_mismatch = {
                    "event": index,
                    "action": token,
                    "detail": "level advanced but goal(step(state, action)) is False",
                }
                break
            else:
                explained += 1
            state, reason = _ground(module, grid, context)
            if reason is not None and str(event["state"]) != "WIN":
                gap(f"e{index}: cannot ground new level ({reason})")
            continue
        if str(event["state"]) == "GAME_OVER":
            if dead is None:
                gap(f"e{index} {token}: GAME_OVER not modeled (define dead(state))")
            elif _is_unknown(predicted) or predicted is None:
                gap(f"e{index} {token}: GAME_OVER transition unmodeled")
            elif not _call(module, "dead", predicted):
                first_mismatch = {
                    "event": index,
                    "action": token,
                    "detail": "environment reported GAME_OVER but dead(step(state, action)) is falsy",
                }
                break
            else:
                explained += 1
            state = None
            continue
        if predicted is None:
            first_mismatch = {
                "event": index,
                "action": token,
                "detail": "step() returned None (impossible) but the environment accepted the action",
            }
            break
        if _is_unknown(predicted):
            gap(f"e{index} {token}: {_unknown_reason(predicted)}")
            state, reason = _ground(module, grid, context)
            if reason is not None:
                gap(f"e{index}: board after unmodeled move not groundable ({reason})")
            continue
        if render is not None:
            rendered = np.asarray(_call(module, "render", predicted), dtype=np.int16)
            pixel_checks += 1
            if rendered.shape == grid.shape and bool(np.array_equal(rendered, grid)):
                pixel_matches += 1
                explained += 1
                state = predicted
            else:
                differing = (
                    int(np.count_nonzero(rendered != grid))
                    if rendered.shape == grid.shape
                    else -1
                )
                first_mismatch = {
                    "event": index,
                    "action": token,
                    "detail": (
                        f"render(state) differs from the observed board in {differing} cells"
                        if differing >= 0
                        else "render(state) has the wrong shape"
                    ),
                }
                break
            continue
        actual_state, reason = _ground(module, grid, context)
        if actual_state is None:
            gap(f"e{index}: board not groundable for comparison ({reason})")
            state = predicted
            continue
        expected_view = _freeze(_call(module, "observe", predicted))
        actual_view = _freeze(_call(module, "observe", actual_state))
        if expected_view == actual_view:
            explained += 1
            state = predicted
        else:
            first_mismatch = {
                "event": index,
                "action": token,
                "expected": expected_view,
                "actual": actual_view,
                "detail": "observe(step(state, action)) differs from observe(initial(observed board))",
            }
            break

    if first_mismatch is not None:
        status = "MISMATCH"
    elif gaps:
        status = "INCOMPLETE"
    else:
        status = "HISTORY_FIT"
    return {
        "status": status,
        "transitions": transitions,
        "explained": explained,
        "gaps": gaps,
        "first_mismatch": first_mismatch,
        "pixel_checks": pixel_checks,
        "pixel_matches": pixel_matches,
        "final_state": state,
    }


def replay_rules(paths: RunPaths) -> dict[str, Any]:
    if not paths.rules.exists():
        raise ArcSkillError("rules.py does not exist; `arc rules init` creates a template")
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    with import_path(paths.rules, "rules") as module:
        check_contract(module)
        result = _walk(module, events)
    result.pop("final_state", None)
    result.update(
        event=int(events[-1]["id"]),
        rules_hash=rules_hash(paths),
        observation_hash=observation_hash(frame_at(events[-1])),
        checked_at=now_iso(),
    )
    return result


def _marker(module: Any, state: Any) -> str:
    key = _optional(module, "key")
    value = key(state) if key is not None else state
    return json.dumps(_freeze(value), sort_keys=True, default=repr)


def solve_rules(
    paths: RunPaths, *, seconds: float = 15.0, max_nodes: int = 250_000
) -> dict[str, Any]:
    if not paths.rules.exists():
        raise ArcSkillError("rules.py does not exist; `arc rules init` creates a template")
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    head = events[-1]
    if str(head["state"]) == "WIN":
        raise ArcSkillError("the game is already complete")
    with import_path(paths.rules, "rules") as module:
        check_contract(module)
        replay = _walk(module, events)
        if replay["status"] == "MISMATCH":
            raise ArcSkillError(
                "rules.py contradicts recorded history at "
                f"e{replay['first_mismatch']['event']} ({replay['first_mismatch']['detail']}); "
                "fix the rules before searching"
            )
        state = replay.pop("final_state")
        if state is None:
            raise ArcSkillError(
                "rules.py cannot ground the current board (initial() returned Unknown); "
                "extend initial() first"
            )

        heuristic = _optional(module, "heuristic")
        dead = _optional(module, "dead")
        started = time.monotonic()
        deadline = started + max(0.1, seconds)
        counter = 0
        nodes = 0
        unknown_edges = 0
        dead_pruned = 0
        start_marker = _marker(module, state)
        best: dict[str, float] = {start_marker: 0.0}
        estimate = float(_call(module, "heuristic", state)) if heuristic else 0.0
        frontier: list[tuple[float, int, float, Any, tuple[str, ...]]] = [
            (estimate, counter, 0.0, state, ())
        ]
        status = "NO_PLAN_IN_MODEL"
        plan_actions: list[str] | None = None
        while frontier:
            if time.monotonic() > deadline:
                status = "TIME_LIMIT"
                break
            if nodes >= max_nodes:
                status = "NODE_LIMIT"
                break
            _, _, cost, current, route = heapq.heappop(frontier)
            if cost != best.get(_marker(module, current)):
                continue
            nodes += 1
            if _call(module, "goal", current):
                status = "PLAN_FOUND"
                plan_actions = list(route)
                break
            for token in _call(module, "actions", current):
                token = str(token).upper()
                successor = _call(module, "step", current, token)
                if successor is None:
                    continue
                if _is_unknown(successor):
                    unknown_edges += 1
                    continue
                if dead is not None and _call(module, "dead", successor):
                    dead_pruned += 1
                    continue
                next_cost = cost + 1.0
                marker = _marker(module, successor)
                if next_cost >= best.get(marker, float("inf")):
                    continue
                best[marker] = next_cost
                counter += 1
                remaining = (
                    float(_call(module, "heuristic", successor)) if heuristic else 0.0
                )
                heapq.heappush(
                    frontier,
                    (next_cost + remaining, counter, next_cost, successor, route + (token,)),
                )
        else:
            status = "NO_PLAN_IN_MODEL"

        result: dict[str, Any] = {
            "status": status,
            "nodes": nodes,
            "frontier": len(frontier),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "unknown_edges": unknown_edges,
            "dead_pruned": dead_pruned,
            "replay_status": replay["status"],
            "replay_gaps": replay["gaps"],
        }
        if plan_actions is None:
            return result

        render = _optional(module, "render")
        predictions: list[dict[str, Any]] = []
        cursor = state
        for position, token in enumerate(plan_actions):
            cursor = _call(module, "step", cursor, token)
            if cursor is None or _is_unknown(cursor):
                raise ArcSkillError(
                    f"search produced an unmodeled step at position {position}; this is a bug in rules.py determinism"
                )
            if position == len(plan_actions) - 1:
                predictions.append({"action": token, "level_up": True})
            else:
                prediction: dict[str, Any] = {
                    "action": token,
                    "observe": _freeze(_call(module, "observe", cursor)),
                }
                if render is not None:
                    prediction["rows"] = grid_to_rows(
                        np.asarray(_call(module, "render", cursor), dtype=np.int16)
                    )
                predictions.append(prediction)

    plan = {
        "kind": "solve-plan",
        "created_at": now_iso(),
        "source": {
            "event": int(head["id"]),
            "level": int(head["levels_completed"]) + 1,
            "observation_hash": observation_hash(frame_at(head)),
            "rules_hash": rules_hash(paths),
        },
        "actions": plan_actions,
        "predictions": predictions,
        "search": {"nodes": nodes, "unknown_edges": unknown_edges},
    }
    atomic_json(paths.plan, plan)
    result["actions"] = plan_actions
    result["plan"] = str(paths.plan)
    return result
