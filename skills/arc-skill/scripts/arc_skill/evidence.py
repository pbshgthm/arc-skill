from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import ArcSkillError, RunPaths, canonical_action, frame_at, load_events

# A stable, high-contrast rendering palette. The numeric grid remains ground truth.
PALETTE = np.asarray(
    [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (255, 255, 255),
        (85, 85, 85),
        (177, 13, 201),
        (127, 255, 212),
        (57, 204, 204),
        (255, 110, 199),
    ],
    dtype=np.uint8,
)


def observation_hash(grid: np.ndarray) -> str:
    value = np.asarray(grid, dtype=np.int16)
    return hashlib.sha256(value.tobytes()).hexdigest()


def image_path(paths: RunPaths, event_id: int, frame_id: int | None = None) -> Path:
    suffix = "settled" if frame_id is None else f"frame-{frame_id:03d}"
    return paths.state / "images" / f"event-{event_id:05d}-{suffix}.png"


def render_grid(grid: np.ndarray, destination: Path, *, scale: int = 8) -> Path:
    array = np.asarray(grid, dtype=np.int16)
    if array.ndim != 2 or (
        array.size and (int(array.min()) < 0 or int(array.max()) > 15)
    ):
        raise ArcSkillError("cannot render a non-ARC grid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(PALETTE[array].astype(np.uint8), mode="RGB")
    image = image.resize(
        (array.shape[1] * scale, array.shape[0] * scale), Image.Resampling.NEAREST
    )
    image.save(destination)
    return destination.resolve()


def render_event(
    paths: RunPaths, event: Mapping[str, Any], *, all_frames: bool = False
) -> list[Path]:
    event_id = int(event["id"])
    output = [render_grid(frame_at(event), image_path(paths, event_id))]
    if all_frames:
        output.extend(
            render_grid(frame_at(event, index), image_path(paths, event_id, index))
            for index in range(len(event["frames"]))
        )
    return output


def history_lines(events: Sequence[Mapping[str, Any]], count: int = 8) -> list[str]:
    paid = 0
    paid_at: dict[int, int] = {}
    for event in events:
        if event.get("counts_action"):
            paid += 1
        paid_at[int(event["id"])] = paid
    lines: list[str] = []
    for event in events[-max(1, count) :]:
        grid = frame_at(event)
        previous = frame_at(events[int(event["id"]) - 1]) if int(event["id"]) else None
        changed = (
            "start"
            if previous is None or previous.shape != grid.shape
            else f"{int(np.count_nonzero(previous != grid))} cells"
        )
        if event.get("predict_ok") is True:
            mark = " ✓"
        elif event.get("predict_ok") is False:
            mark = " ✗"
        else:
            mark = ""
        lines.append(
            f"  e{int(event['id']):04d} a{paid_at[int(event['id'])]:04d} "
            f"L{min(int(event['win_levels']), int(event['levels_completed']) + 1)} {canonical_action(event)}{mark} | {changed} | "
            f"frames={len(event['frames'])} | {event['state']}"
        )
    return lines


def current_image(paths: RunPaths) -> Path:
    events = load_events(paths)
    if not events:
        raise ArcSkillError("timeline is empty")
    destination = image_path(paths, int(events[-1]["id"]))
    if not destination.exists():
        render_event(paths, events[-1])
    return destination.resolve()
