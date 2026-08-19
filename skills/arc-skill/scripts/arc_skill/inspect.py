from __future__ import annotations

from collections import Counter
import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    ArcSkillError,
    RunPaths,
    canonical_action,
    frame_at,
    load_events,
    load_jsonl,
    read_json,
)
from .evidence import (
    current_image,
    history_lines,
    observation_hash,
    render_event,
)
from .perception import (
    connected_components,
    infer_lattice,
    motion_trace,
    repeated_shapes,
    transition_story,
)
from .rules import rules_hash

ACTION_PRIORS = {
    1: "up / interface direction 1",
    2: "down / interface direction 2",
    3: "left / interface direction 3",
    4: "right / interface direction 4",
    5: "interact",
    6: "point/click at x,y (x=column, y=row)",
    7: "undo",
}


def _available_line(event: Mapping[str, Any]) -> str:
    return "ACTIONS | " + (
        " · ".join(
            f"ACTION{number}={ACTION_PRIORS.get(number, 'unknown public action')}"
            for number in event["available_actions"]
        )
        or "none"
    )


def _bbox(mask: np.ndarray, margin: int = 1) -> tuple[int, int, int, int]:
    cells = np.argwhere(mask)
    height, width = mask.shape
    if not len(cells):
        return 0, height, 0, width
    top = max(0, int(cells[:, 0].min()) - margin)
    bottom = min(height, int(cells[:, 0].max()) + margin + 1)
    left = max(0, int(cells[:, 1].min()) - margin)
    right = min(width, int(cells[:, 1].max()) + margin + 1)
    return top, bottom, left, right


def _grid_text(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> str:
    top, bottom, left, right = bounds
    rows = [f"     cols {left}..{right - 1}"]
    rows.extend(
        f"{row:>3}  "
        + "".join(format(int(cell), "x") for cell in grid[row, left:right])
        for row in range(top, bottom)
    )
    return "\n".join(rows)


def _masked_grid_text(
    grid: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int]
) -> str:
    top, bottom, left, right = bounds
    lines = [f"     cols {left}..{right - 1}"]
    active = [row for row in range(top, bottom) if bool(mask[row, left:right].any())]
    last: int | None = None
    for row in active:
        if last is not None and row - last > 1:
            lines.append(f"      ⋮ {row - last - 1} unchanged rows")
        lines.append(
            f"{row:>3}  "
            + "".join(
                format(int(grid[row, col]), "x") if mask[row, col] else "·"
                for col in range(left, right)
            )
        )
        last = row
    return "\n".join(lines)


def _salient_components(grid: np.ndarray, limit: int = 12) -> list[dict[str, Any]]:
    """Rank neutral click hypotheses without flooding the output with pixel noise."""
    components = connected_components(grid)
    counts = np.bincount(grid.ravel(), minlength=16)
    background = int(np.argmax(counts))
    height, width = grid.shape
    shape_counts = Counter(
        (
            item["color"],
            item["shape_key"][0],
            item["shape_key"][1],
            item["shape_key"][2],
        )
        for item in components
    )
    candidates = []
    for item in components:
        if item["color"] == background or item["size"] < 2:
            continue
        if item["size"] > grid.size // 4:
            continue
        row0, row1, col0, col1 = item["bbox"]
        box_height, box_width = row1 - row0 + 1, col1 - col0 + 1
        # Long screen-edge strips are usually chrome or framing, not a useful
        # representative click point. They remain visible in the board itself.
        frame_like = (box_height >= height // 2 or box_width >= width // 2) and (
            row0 == 0 or col0 == 0 or row1 == height - 1 or col1 == width - 1
        )
        if frame_like:
            continue
        key = (item["color"], box_height, box_width, item["shape_key"][2])
        density = item["size"] / max(1, box_height * box_width)
        candidates.append(
            {
                **item,
                "repeat_count": shape_counts[key],
                "density": density,
            }
        )
    candidates.sort(
        key=lambda item: (
            -min(int(item["repeat_count"]), 8),
            -min(int(item["size"]), 256),
            -float(item["density"]),
            item["bbox"],
        )
    )
    return candidates[:limit]


def _scene_summary(grid: np.ndarray) -> list[str]:
    counts = Counter(int(value) for value in grid.ravel())
    components = connected_components(grid)
    repeated = repeated_shapes(grid, min_size=2, limit=5)
    colors = " ".join(f"{color:x}:{count}" for color, count in counts.most_common(8))
    lines = [
        f"SCENE | {grid.shape[0]}x{grid.shape[1]} | colors {colors} | monochrome components {len(components)}",
    ]
    lattice = infer_lattice(grid).get("candidates", ())
    if lattice:
        guesses = []
        for item in lattice[:3]:
            origin = item.get("origin", [0, 0])
            guesses.append(
                f"scale {item['scale']} origin(row={origin[0]},col={origin[1]}) support={item['support_fraction']:.2f}"
            )
        lines.append("LATTICE HYPOTHESES | " + " · ".join(guesses))
    for item in repeated:
        centers = " ".join(
            f"({center[1]:g},{center[0]:g})" for center in item.get("centers", ())[:6]
        )
        lines.append(
            f"REPEATED SHAPE | color={item['color']:x} size={item['size']} shape={item['shape'][0]}x{item['shape'][1]} count={item['count']} centers(x,y)={centers}"
        )
    return lines


def _bounded(items: list[str], limit: int, *, preserve_ends: bool = False) -> list[str]:
    if len(items) <= limit:
        return items
    omitted = len(items) - limit
    if preserve_ends and limit >= 2:
        left = limit // 2
        right = limit - left
        return items[:left] + [f"  … {omitted} more"] + items[-right:]
    return [f"  … {omitted} earlier"] + items[-limit:]


def _level_action_count(events: Sequence[Mapping[str, Any]]) -> int:
    completed = int(events[-1]["levels_completed"])
    count = 0
    for event in reversed(events):
        if int(event["levels_completed"]) != completed:
            break
        if event.get("counts_action"):
            count += 1
    return count


def _recent_predictions(
    events: Sequence[Mapping[str, Any]], window: int = 10
) -> tuple[int, int]:
    graded = [event for event in events if event.get("predict_ok") is not None]
    recent = graded[-window:]
    hits = sum(1 for event in recent if event["predict_ok"])
    return hits, len(recent)


def _nudges(events: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    level_actions = _level_action_count(events)
    hits, total = _recent_predictions(events)
    misses = total - hits
    if level_actions >= 40:
        lines.append(
            f"NUDGE | {level_actions} paid actions on this level — stop manual probing; "
            "write rules.py for it (`arc rules help`), verify with `arc rules replay`, "
            "then `arc rules solve`"
        )
    elif level_actions >= 25:
        lines.append(
            f"NUDGE | {level_actions} paid actions on this level — re-read your notes, "
            "kill dead assumptions, and consider the rules.py tier (`arc rules help`)"
        )
    if misses >= 3:
        lines.append(
            f"NUDGE | {misses} of the last {total} predictions missed — the mechanics "
            "story in NOTES.md is wrong; fix it before spending more actions"
        )
    return lines


def _rules_lines(paths: RunPaths, event: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    current_hash = rules_hash(paths)
    if current_hash is None:
        return lines
    verification = read_json(paths.verification, {})
    fresh = (
        isinstance(verification, dict)
        and verification.get("event") == int(event["id"])
        and verification.get("rules_hash") == current_hash
        and verification.get("observation_hash") == observation_hash(frame_at(event))
    )
    if fresh:
        status = verification.get("status")
        gaps = verification.get("gaps") or []
        summary = (
            f"RULES | {status} | {verification.get('explained', 0)}/"
            f"{verification.get('transitions', 0)} transitions explained"
        )
        if gaps:
            summary += f" | {len(gaps)} gaps"
        lines.append(summary)
        mismatch = verification.get("first_mismatch")
        if status == "MISMATCH" and isinstance(mismatch, dict):
            lines.append(
                f"  first mismatch e{mismatch.get('event')}: {mismatch.get('detail')}"
            )
        lines.extend(f"  gap: {gap}" for gap in gaps[:3])
    else:
        lines.append(
            "RULES | rules.py present, replay unchecked or stale — run `arc rules replay`"
        )
    plan = read_json(paths.plan, {})
    if isinstance(plan, dict) and plan.get("kind") == "solve-plan":
        source = plan.get("source", {})
        checks = (
            ("event", int(event["id"])),
            ("observation_hash", observation_hash(frame_at(event))),
            ("rules_hash", current_hash),
        )
        stale = [name for name, expected in checks if source.get(name) != expected]
        if stale:
            lines.append(
                f"PLAN | stale ({', '.join(stale)} changed) — rerun `arc rules solve`"
            )
        else:
            lines.append(
                f"PLAN | fresh | {len(plan.get('actions', ()))} actions — execute with "
                "`arc commit @.arc/plan.json`"
            )
    return lines


def _demotion_banner(paths: RunPaths, event: Mapping[str, Any]) -> list[str]:
    completed = int(event["levels_completed"])
    if completed <= 0 or str(event["state"]) == "WIN":
        return []
    archive = paths.state / "levels" / f"level-{completed}.md"
    if not archive.exists() or not paths.notes.exists():
        return []
    if paths.notes.stat().st_mtime <= archive.stat().st_mtime:
        return [
            f"NOTES | unchanged since level {completed} ended — earlier Verified claims "
            "are only Assumed on this level until re-tested"
        ]
    return []


def _notes_lines(paths: RunPaths) -> list[str]:
    try:
        text = paths.notes.read_text()
    except FileNotFoundError:
        return ["NOTES | missing — create .arc/NOTES.md and keep it current"]
    content = [f"  {line[:240]}" for line in text.splitlines()]
    return [f"NOTES | {paths.notes} (edit the file directly; shown in full)"] + _bounded(
        content, 120, preserve_ends=True
    )


def _mode_line(paths: RunPaths) -> str:
    config = read_json(paths.config, {})
    mode = str(config.get("mode", "local")) if isinstance(config, dict) else "local"
    if mode != "competition":
        return (
            "MODE | LOCAL SIMULATOR | competition action/reset accounting | "
            "exact replay recovery enabled"
        )
    mutations = load_jsonl(paths.mutations)
    last = (
        mutations[-1].get("timestamp")
        if mutations
        else (config.get("created_at") if isinstance(config, dict) else None)
    )
    lease = "unknown"
    if last:
        try:
            then = dt.datetime.fromisoformat(str(last))
            if then.tzinfo is None:
                then = then.replace(tzinfo=dt.timezone.utc)
            idle = max(
                0.0,
                (dt.datetime.now(dt.timezone.utc) - then).total_seconds(),
            )
            lease = (
                "expired/unavailable"
                if idle >= 15 * 60
                else f"about {max(0, 15 - int(idle // 60))}m action-idle remaining"
            )
        except ValueError:
            pass
    return f"MODE | REMOTE COMPETITION | {lease} | exact replay recovery unavailable"


def result_text(paths: RunPaths, receipt: Mapping[str, Any]) -> str:
    """Self-sufficient printout after a paid command: outcome, grade, new board."""
    events = load_events(paths)
    event = events[-1]
    lines = [f"OUTCOME | {receipt['outcome']} | {receipt['detail']}"]
    lines.extend(f"  {line}" for line in receipt.get("grade", ()))
    for step in receipt.get("steps", ()):
        mark = "✓" if step.get("ok") else "✗"
        lines.append(f"  e{int(step['event']):04d} {step['action']} {mark}")
        lines.extend(f"      {failure}" for failure in step.get("failed", ()))
        if step.get("problem"):
            lines.append(f"      {step['problem']}")
    paid = sum(bool(item.get("counts_action")) for item in events)
    lines.append(
        f"EVENT | e{int(event['id'])} | level "
        f"{min(int(event['win_levels']), int(event['levels_completed']) + 1)}"
        f"/{event['win_levels']} | paid actions {paid} | {event['state']}"
    )
    end = receipt.get("end_event")
    if end is not None and int(event["id"]) == int(end) and int(end) > 0:
        previous = frame_at(events[int(end) - 1])
        current = frame_at(event)
        story = transition_story(previous, current)
        lines.append("TRANSITION | last step, worded")
        lines.extend(f"  {item}" for item in story["lines"])
        if previous.shape == current.shape:
            mask = previous != current
            box = _bbox(mask, margin=1)
            # Cell-exact compare inline only while it stays compact; `arc view`
            # renders the same DIFF for bigger transitions.
            if bool(mask.any()) and int(mask.sum()) <= 200 and box[1] - box[0] <= 18:
                lines.extend(
                    [
                        "DIFF | before — changed cells only ('·' unchanged)",
                        _masked_grid_text(previous, mask, box),
                        "DIFF | after",
                        _masked_grid_text(current, mask, box),
                    ]
                )
    lines.append(f"IMAGE | {current_image(paths)}")
    lines.append(_available_line(event))
    return "\n".join(lines)


def inspect_text(
    paths: RunPaths,
    *,
    event_id: int | None = None,
    full: bool = False,
    frames: bool = False,
) -> str:
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    index = len(events) - 1 if event_id is None else event_id
    if not 0 <= index < len(events):
        raise ArcSkillError(f"event must be in 0..{len(events) - 1}")
    event = events[index]
    grid = frame_at(event)
    previous = frame_at(events[index - 1]) if index else None
    lines = [
        f"RUN | event {index} | level {min(int(event['win_levels']), int(event['levels_completed']) + 1)}/{event['win_levels']} | paid actions {sum(bool(item.get('counts_action')) for item in events)} | state {event['state']}",
        f"CAUSE | {canonical_action(event)} | frames {len(event['frames'])}",
        _available_line(event),
    ]
    changed: np.ndarray | None = None
    if previous is not None and previous.shape == grid.shape:
        changed = previous != grid
        bounds = (
            (0, grid.shape[0], 0, grid.shape[1]) if full else _bbox(changed, margin=2)
        )
        lines.append(
            f"SETTLED DIFF | {int(changed.sum())} cells | crop rows {bounds[0]}..{bounds[1] - 1}, cols {bounds[2]}..{bounds[3] - 1}"
        )
    else:
        bounds = (0, grid.shape[0], 0, grid.shape[1])
    if previous is not None:
        story = transition_story(previous, grid)
        lines.append("TRANSITION | since previous settled board")
        lines.extend(f"  {item}" for item in story["lines"])
    if changed is not None and not full and not bool(changed.any()):
        lines.append("BOARD | settled frame unchanged; use --grid to redraw it")
    else:
        lines.extend(["BOARD | color indices 0-f", _grid_text(grid, bounds)])

    if (
        changed is not None
        and previous is not None
        and bool(changed.any())
        and int(changed.sum()) <= changed.size * 3 // 5
    ):
        diff_bounds = _bbox(changed, margin=1)
        lines.extend(
            [
                "DIFF | before — changed cells only ('·' unchanged)",
                _masked_grid_text(previous, changed, diff_bounds),
                "DIFF | after",
                _masked_grid_text(grid, changed, diff_bounds),
            ]
        )

    lines.extend(_scene_summary(grid))

    if len(event["frames"]) > 1:
        lines.append("ANIMATION | consecutive causal deltas")
        prior_frame = (
            previous
            if previous is not None and previous.shape == grid.shape
            else frame_at(event, 0)
        )
        animation_frames = [prior_frame] + [
            frame_at(event, index) for index in range(len(event["frames"]))
        ]
        traces = motion_trace(animation_frames)
        for frame_index, trace in enumerate(traces):
            delta = trace["delta"]
            box = delta.get("bbox")
            region = (
                "none"
                if box is None
                else f"rows {box[0]}..{box[1]}, cols {box[2]}..{box[3]}"
            )
            changes = ", ".join(
                f"{item['from']:x}→{item['to']:x}×{item['count']}"
                for item in delta.get("color_changes", ())[:3]
            )
            lines.append(
                f"  frame {frame_index + 1}/{len(event['frames'])}: {delta['changed_cells']} changed; {region}"
                + (f"; {changes}" if changes else "")
            )
            for moved in trace.get("moved", ())[:3]:
                before_center, after_center = moved["from"], moved["to"]
                lines.append(
                    f"    MOVED | color={moved['color']:x} size={moved['size']} "
                    f"({before_center[1]:g},{before_center[0]:g})→({after_center[1]:g},{after_center[0]:g}) x,y"
                )
            if frames:
                current = animation_frames[frame_index + 1]
                lines.append(
                    _grid_text(current, (0, current.shape[0], 0, current.shape[1]))
                )

    if 6 in event["available_actions"]:
        candidates = _salient_components(grid)
        lines.append(
            "CLICK CANDIDATES | salience-ranked monochrome representatives; hypotheses only"
        )
        for item in candidates:
            top, bottom, left, right = item["bbox"]
            y, x = item["representative"]
            lines.append(
                f"  ACTION6:{x},{y} color={item['color']:x} size={item['size']} repeats={item['repeat_count']} "
                f"bbox=row {top}..{bottom}, col {left}..{right}"
            )
    return "\n".join(lines)


def status_text(paths: RunPaths, *, history: int = 8) -> str:
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    event = events[-1]
    images = render_event(paths, event)
    paid = sum(bool(item.get("counts_action")) for item in events)
    config = read_json(paths.config, {})
    game_id = (
        str(config.get("game_id", "unknown")) if isinstance(config, dict) else "unknown"
    )
    level_actions = _level_action_count(events)
    hits, total = _recent_predictions(events)
    prediction_summary = (
        f"predictions {hits}/{total} ✓ over the last {total}"
        if total
        else "no graded predictions yet"
    )
    lines = [
        f"STATUS | {game_id} | event {event['id']} | level {min(int(event['win_levels']), int(event['levels_completed']) + 1)}/{event['win_levels']} | paid actions {paid} | {event['state']}",
        _mode_line(paths),
        f"IMAGE | {images[0]}",
        _available_line(event),
        f"LEVEL | {level_actions} paid actions this level | {prediction_summary}",
        *_rules_lines(paths, event),
        *_nudges(events),
        "RECENT | ✓ prediction held · ✗ prediction missed",
        *history_lines(events, history),
        *_demotion_banner(paths, event),
        *_notes_lines(paths),
    ]
    return "\n".join(lines)


def view_text(
    paths: RunPaths,
    *,
    event_id: int | None = None,
    grid: bool = False,
    frames: bool = False,
    crop: tuple[int, int, int, int] | None = None,
    history: int = 0,
) -> str:
    events = load_events(paths)
    index = len(events) - 1 if event_id is None else event_id
    if not 0 <= index < len(events):
        raise ArcSkillError(f"event must be in 0..{len(events) - 1}")
    event = events[index]
    images = render_event(paths, event, all_frames=frames)
    lines = [
        f"IMAGE | {images[0]}",
        inspect_text(paths, event_id=index, full=grid, frames=frames),
    ]
    if frames and len(images) > 1:
        lines.append("FRAME IMAGES | " + " ".join(str(path) for path in images[1:]))
    if crop is not None:
        array = frame_at(event)
        top, bottom, left, right = crop
        if not (
            0 <= top < bottom <= array.shape[0] and 0 <= left < right <= array.shape[1]
        ):
            raise ArcSkillError(
                f"crop must fit rows 0..{array.shape[0] - 1}, cols 0..{array.shape[1] - 1}"
            )
        lines.extend(
            [
                f"EXACT CROP | rows {top}..{bottom - 1}, cols {left}..{right - 1}",
                _grid_text(array, crop),
            ]
        )
    if history:
        lines.append("HISTORY | cause → observed result")
        lines.extend(history_lines(events[: index + 1], history))
    return "\n".join(lines)


def export_history(paths: RunPaths, destination: Path) -> Path:
    if destination.suffix.lower() != ".npz":
        raise ArcSkillError("observation export filename must end in .npz")
    try:
        destination.resolve().relative_to(paths.root.resolve())
    except ValueError as error:
        raise ArcSkillError(
            "observation export must stay inside the run directory"
        ) from error
    try:
        destination.resolve().relative_to(paths.state.resolve())
    except ValueError:
        pass
    else:
        raise ArcSkillError("do not write analysis output inside .arc")
    events = load_events(paths)
    frames: list[np.ndarray] = []
    frame_events: list[int] = []
    for event in events:
        for index in range(len(event["frames"])):
            frames.append(frame_at(event, index))
            frame_events.append(int(event["id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        frames=np.stack(frames),
        frame_events=np.asarray(frame_events, dtype=np.int32),
        settled=np.stack([frame_at(event) for event in events]),
        actions=np.asarray([canonical_action(event) for event in events]),
        levels=np.asarray(
            [event["levels_completed"] for event in events], dtype=np.int16
        ),
        states=np.asarray([event["state"] for event in events]),
        available=np.asarray(
            [
                [number in event["available_actions"] for number in range(1, 8)]
                for event in events
            ],
            dtype=bool,
        ),
    )
    return destination.resolve()
