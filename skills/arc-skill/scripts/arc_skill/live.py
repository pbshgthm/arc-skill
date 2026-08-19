"""Paid-action execution: act with a graded prediction, batch with per-step
claims, execute a rules.py solve-plan, and reset with a reason.

Every paid action is journaled by the broker before the response is recorded,
so a crash between spend and record is recovered on the next start.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .broker import broker_step
from .core import (
    ArcSkillError,
    RunPaths,
    append_event,
    append_jsonl,
    atomic_json,
    canonical_action,
    check_public_action,
    context_for,
    frame_at,
    import_path,
    load_events,
    make_event,
    now_iso,
    parse_action,
    read_json,
    rows_to_grid,
)
from .evidence import observation_hash, render_event
from .perception import build_scene_dossier
from .predictions import grade_claims, grade_lines, parse_claims
from .rules import (
    _call,
    _freeze,
    _is_unknown,
    _unknown_reason,
    check_contract,
    rules_hash,
)


def _receipt(paths: RunPaths, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {"timestamp": now_iso(), **dict(value)}
    paths.receipts.mkdir(parents=True, exist_ok=True)
    event = record.get("end_event", record.get("start_event", "none"))
    atomic_json(paths.receipts / f"{time.time_ns()}-e{event}.json", record)
    append_jsonl(paths.activity, {"kind": "receipt", **record})
    return record


def _archive_notes(paths: RunPaths, completed_level: int) -> None:
    """Snapshot the live notes when a level completes; the live file is kept."""
    if not paths.notes.exists():
        return
    archive = paths.state / "levels" / f"level-{completed_level}.md"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_text(paths.notes.read_text())


def _head_events(paths: RunPaths, at_event: int | None) -> list[dict[str, Any]]:
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    if events[-1]["state"] == "WIN":
        raise ArcSkillError("the game is already complete")
    if at_event is not None and at_event != int(events[-1]["id"]):
        raise ArcSkillError(
            f"event guard failed: requested {at_event}, current {events[-1]['id']}"
        )
    return events


def _paid_step(
    paths: RunPaths,
    token: str,
    reasoning: Mapping[str, Any] | None = None,
    note: str = "",
) -> tuple[dict[str, Any], np.ndarray, str | None]:
    prior = load_events(paths)[-1]
    name, data = parse_action(token)
    if name == "RESET":
        raise ArcSkillError("use `arc reset`; reset cannot hide inside a batch")
    check_public_action(token, prior["available_actions"])
    response, mutation_id, warning = broker_step(paths, name, data, reasoning)
    pending = make_event(response, name, data, prior, note=note)
    pending["mutation_id"] = mutation_id
    return pending, frame_at(prior), warning


def _record(paths: RunPaths, pending: Mapping[str, Any]) -> dict[str, Any]:
    event = append_event(paths, pending)
    render_event(paths, event)
    build_scene_dossier(paths, load_events(paths))
    level_before = event.get("level_before")
    if level_before is not None and int(event["levels_completed"]) > int(level_before):
        _archive_notes(paths, int(event["levels_completed"]))
    return event


def _level_advanced(event: Mapping[str, Any]) -> bool:
    level_before = event.get("level_before")
    return level_before is not None and int(event["levels_completed"]) > int(
        level_before
    )


def execute_action(
    paths: RunPaths,
    token: str,
    *,
    predict: str,
    because: str | None = None,
    at_event: int | None = None,
) -> dict[str, Any]:
    claims = parse_claims(predict)
    events = _head_events(paths, at_event)
    start_event = int(events[-1]["id"])
    reasoning: dict[str, Any] = {"predict": predict}
    if because:
        reasoning["because"] = because
    pending, before, warning = _paid_step(paths, token, reasoning, note=because or "")
    graded = grade_claims(claims, before, pending)
    ok = all(item["ok"] for item in graded)
    pending["predict"] = predict
    pending["predict_ok"] = ok
    pending["grade"] = graded
    event = _record(paths, pending)
    lines = grade_lines(graded)
    if event["state"] == "WIN":
        outcome, detail = "GAME_COMPLETE", "the game is complete"
    elif _level_advanced(event):
        outcome = "LEVEL_COMPLETE"
        detail = (
            f"level {int(event['level_before']) + 1} complete; notes archived — "
            "re-verify carried assumptions on the new board"
        )
        if not ok:
            detail += "; prediction also missed — treat the mechanics as unproven"
    elif event["state"] == "GAME_OVER":
        outcome = "GAME_OVER"
        detail = "environment reported GAME_OVER; `arc reset` restarts the level"
    elif not ok:
        outcome = "SURPRISE"
        first_failed = next(line for line in lines if line.startswith("✗"))
        detail = f"prediction missed: {first_failed[2:]}"
    else:
        outcome, detail = "PREDICTED", "result matched the prediction"
    if warning:
        detail += f"; scorecard finalization warning: {warning}"
    return _receipt(
        paths,
        {
            "kind": "act",
            "outcome": outcome,
            "detail": detail,
            "start_event": start_event,
            "end_event": int(event["id"]),
            "level": min(
                int(event["win_levels"]), int(event["levels_completed"]) + 1
            ),
            "action": canonical_action(event),
            "predict": predict,
            "grade": lines,
            "because": because,
        },
    )


def parse_step(raw: str) -> tuple[str, str]:
    action, separator, predict = raw.partition("::")
    if not separator or not action.strip() or not predict.strip():
        raise ArcSkillError(
            'each step needs its own prediction: --step "ACTION1 :: <claims>"'
        )
    return action.strip().upper(), predict.strip()


def _validate_batch_tokens(tokens: Sequence[str]) -> None:
    """Reject a bad token before any step spends an action."""
    for token in tokens:
        name, _ = parse_action(token)
        if name == "RESET":
            raise ArcSkillError("use `arc reset`; reset cannot hide inside a batch")


def execute_steps(
    paths: RunPaths,
    raw_steps: Sequence[str],
    *,
    at_event: int | None = None,
) -> dict[str, Any]:
    if not raw_steps:
        raise ArcSkillError("no steps supplied")
    parsed: list[tuple[str, str, list[dict[str, Any]]]] = []
    for raw in raw_steps:
        token, predict = parse_step(raw)
        parsed.append((token, predict, parse_claims(predict)))
    _validate_batch_tokens([token for token, _, _ in parsed])
    events = _head_events(paths, at_event)
    start_event = int(events[-1]["id"])
    records: list[dict[str, Any]] = []
    outcome = "PREDICTED"
    detail = f"all {len(parsed)} steps landed as predicted"
    last_warning: str | None = None
    for index, (token, predict, claims) in enumerate(parsed):
        pending, before, warning = _paid_step(paths, token, {"predict": predict})
        last_warning = warning or last_warning
        graded = grade_claims(claims, before, pending)
        ok = all(item["ok"] for item in graded)
        pending["predict"] = predict
        pending["predict_ok"] = ok
        pending["grade"] = graded
        event = _record(paths, pending)
        failed = [line for line in grade_lines(graded) if line.startswith("✗")]
        records.append(
            {
                "event": int(event["id"]),
                "action": canonical_action(event),
                "ok": ok,
                "failed": failed,
            }
        )
        remaining = len(parsed) - index - 1
        discarded = f"; {remaining} remaining steps were discarded" if remaining else ""
        if event["state"] == "WIN":
            outcome, detail = "GAME_COMPLETE", f"the game is complete{discarded}"
            break
        if _level_advanced(event):
            outcome = "LEVEL_COMPLETE"
            detail = f"level advanced after {canonical_action(event)}{discarded}"
            if not ok:
                detail += "; prediction also missed — treat the mechanics as unproven"
            break
        if event["state"] == "GAME_OVER":
            outcome = "GAME_OVER"
            detail = (
                f"environment reported GAME_OVER after {canonical_action(event)}"
                f"{discarded}"
            )
            break
        if not ok:
            outcome = "SURPRISE"
            detail = f"step {index + 1} missed: {failed[0][2:]}{discarded}"
            break
    if last_warning:
        detail += f"; scorecard finalization warning: {last_warning}"
    return _receipt(
        paths,
        {
            "kind": "commit",
            "outcome": outcome,
            "detail": detail,
            "start_event": start_event,
            "end_event": int(load_events(paths)[-1]["id"]),
            "steps": records,
        },
    )


def _load_solve_plan(paths: RunPaths, reference: str) -> dict[str, Any]:
    candidate = Path(reference[1:] if reference.startswith("@") else reference)
    path = candidate if candidate.is_absolute() else paths.root / candidate
    try:
        path.resolve().relative_to(paths.root.resolve())
    except ValueError as error:
        raise ArcSkillError(
            "plan reference must stay inside the run directory"
        ) from error
    value = read_json(path)
    if not isinstance(value, dict) or value.get("kind") != "solve-plan":
        raise ArcSkillError(
            "commit accepts only a solve-plan written by `arc rules solve`"
        )
    actions = value.get("actions")
    predictions = value.get("predictions")
    if (
        not isinstance(actions, list)
        or not actions
        or not isinstance(predictions, list)
        or len(predictions) != len(actions)
    ):
        raise ArcSkillError(
            "plan needs one prediction per action; rerun `arc rules solve`"
        )
    return value


def execute_solve_plan(
    paths: RunPaths,
    reference: str,
    *,
    at_event: int | None = None,
) -> dict[str, Any]:
    events = _head_events(paths, at_event)
    plan = _load_solve_plan(paths, reference)
    latest = events[-1]
    current = {
        "event": int(latest["id"]),
        "observation_hash": observation_hash(frame_at(latest)),
        "rules_hash": rules_hash(paths),
    }
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ArcSkillError("plan has no source provenance; rerun `arc rules solve`")
    stale = [name for name, expected in current.items() if source.get(name) != expected]
    if stale:
        raise ArcSkillError(
            f"plan is stale ({', '.join(stale)} changed); rerun `arc rules solve`"
        )
    start_event = int(latest["id"])
    actions = [str(item).upper() for item in plan["actions"]]
    _validate_batch_tokens(actions)
    predictions = plan["predictions"]
    records: list[dict[str, Any]] = []
    outcome = "PREDICTED"
    detail = f"all {len(actions)} steps landed as predicted"
    last_warning: str | None = None
    with import_path(paths.rules, "rules") as module:
        check_contract(module)
        for index, (token, expected) in enumerate(zip(actions, predictions)):
            pending, _, warning = _paid_step(
                paths, token, {"plan": reference, "plan_step": index}
            )
            last_warning = warning or last_warning
            event = _record(paths, pending)
            advanced = _level_advanced(event)
            ok = False
            problem = ""
            if expected.get("level_up"):
                ok = advanced
                if not ok:
                    problem = (
                        "plan predicted level completion here, but the level "
                        "did not advance"
                    )
            elif advanced:
                problem = (
                    "level advanced earlier than the plan predicted; "
                    "rules.py is wrong somewhere"
                )
            elif "rows" in expected:
                predicted_grid = rows_to_grid(expected["rows"])
                actual = frame_at(event)
                ok = predicted_grid.shape == actual.shape and bool(
                    np.array_equal(predicted_grid, actual)
                )
                if not ok:
                    problem = "board differs from the render(state) prediction"
            else:
                all_events = load_events(paths)
                context = context_for(all_events, len(all_events) - 1)
                grounded = _call(module, "initial", frame_at(event), context)
                if grounded is None or _is_unknown(grounded):
                    reason = (
                        _unknown_reason(grounded)
                        if grounded is not None
                        else "initial() returned None"
                    )
                    problem = f"board after this step is not groundable ({reason})"
                else:
                    actual_view = _freeze(_call(module, "observe", grounded))
                    ok = actual_view == expected.get("observe")
                    if not ok:
                        problem = (
                            "observe(state) differs from the plan's prediction"
                        )
            records.append(
                {
                    "event": int(event["id"]),
                    "action": canonical_action(event),
                    "ok": ok,
                    **({"problem": problem} if problem else {}),
                }
            )
            remaining = len(actions) - index - 1
            discarded = (
                f"; {remaining} remaining steps were discarded" if remaining else ""
            )
            if event["state"] == "WIN":
                outcome, detail = "GAME_COMPLETE", f"the game is complete{discarded}"
                break
            if advanced:
                outcome = "LEVEL_COMPLETE"
                detail = (
                    f"plan completed the level as predicted{discarded}"
                    if ok
                    else f"{problem}{discarded}"
                )
                break
            if event["state"] == "GAME_OVER":
                outcome = "GAME_OVER"
                detail = (
                    "environment reported GAME_OVER during the plan; "
                    f"run `arc rules replay` to find the broken rule{discarded}"
                )
                break
            if not ok:
                outcome = "SURPRISE"
                detail = (
                    f"step {index + 1} ({token}) diverged: {problem}; "
                    f"run `arc rules replay` to find the broken rule{discarded}"
                )
                break
    if last_warning:
        detail += f"; scorecard finalization warning: {last_warning}"
    return _receipt(
        paths,
        {
            "kind": "commit",
            "outcome": outcome,
            "detail": detail,
            "start_event": start_event,
            "end_event": int(load_events(paths)[-1]["id"]),
            "plan": reference,
            "steps": records,
        },
    )


def reset_level(
    paths: RunPaths,
    *,
    because: str | None = None,
    at_event: int | None = None,
) -> dict[str, Any]:
    events = _head_events(paths, at_event)
    prior = events[-1]
    reason = because
    if prior["state"] == "GAME_OVER":
        reason = reason or "environment reported GAME_OVER"
    elif not reason:
        raise ArcSkillError(
            'reset needs --because "<why this board is worth abandoning>" '
            "(GAME_OVER excepted)"
        )
    response, mutation_id, warning = broker_step(
        paths, "RESET", None, {"because": reason}
    )
    pending = make_event(response, "RESET", None, prior, note=reason)
    pending["mutation_id"] = mutation_id
    event = _record(paths, pending)
    detail = "current board rewound; completed levels and action history preserved"
    if warning:
        detail += f"; scorecard finalization warning: {warning}"
    return _receipt(
        paths,
        {
            "kind": "reset",
            "outcome": "RESET",
            "detail": detail,
            "start_event": int(prior["id"]),
            "end_event": int(event["id"]),
            "because": reason,
        },
    )
