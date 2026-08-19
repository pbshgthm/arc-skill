from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import gcd
from typing import Any

import numpy as np

from .core import RunPaths, atomic_json, canonical_action, frame_at


def connected_components(
    grid: np.ndarray,
    *,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Return neutral monochrome 4-connected regions.

    These are geometric candidates, not asserted game objects. Coordinates use
    x=column and y=row everywhere exposed to the agent.
    """
    board = np.asarray(grid)
    seen = np.zeros(board.shape, dtype=bool)
    output: list[dict[str, Any]] = []
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            if seen[row, col]:
                continue
            color = int(board[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(board[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_size:
                continue
            rows = [value[0] for value in cells]
            cols = [value[1] for value in cells]
            row0, row1 = min(rows), max(rows)
            col0, col1 = min(cols), max(cols)
            normalized = tuple(sorted((y - row0, x - col0) for y, x in cells))
            signature = (
                f"{row1 - row0 + 1}x{col1 - col0 + 1}:"
                + ";".join(f"{y},{x}" for y, x in normalized[:40])
                + (";..." if len(normalized) > 40 else "")
            )
            center = [sum(rows) / len(rows), sum(cols) / len(cols)]
            representative = min(
                cells,
                key=lambda value: (
                    abs(value[0] - center[0]) + abs(value[1] - center[1]),
                    value,
                ),
            )
            output.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [row0, row1, col0, col1],
                    "center": [round(center[0], 3), round(center[1], 3)],
                    "representative": [representative[0], representative[1]],
                    "signature": signature,
                    "shape_key": [row1 - row0 + 1, col1 - col0 + 1, normalized],
                }
            )
    output.sort(key=lambda item: (item["size"], item["color"], item["bbox"]))
    return output


def repeated_shapes(
    grid: np.ndarray,
    *,
    min_size: int = 2,
    limit: int = 24,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for component in connected_components(grid, min_size=min_size):
        height, width, normalized = component["shape_key"]
        groups[(component["color"], height, width, normalized)].append(component)
    repeated = []
    for (color, height, width, _), members in groups.items():
        if len(members) < 2:
            continue
        repeated.append(
            {
                "color": color,
                "shape": [height, width],
                "size": members[0]["size"],
                "count": len(members),
                "centers": [item["center"] for item in members[:32]],
                "bboxes": [item["bbox"] for item in members[:32]],
                "signature": members[0]["signature"],
            }
        )
    repeated.sort(key=lambda item: (-item["count"], -item["size"], item["color"]))
    return repeated[:limit]


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if not len(values):
        return []
    output: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for index in range(1, len(values)):
        value = int(values[index])
        if value == current:
            continue
        output.append((start, index - 1, current))
        start, current = index, value
    output.append((start, len(values) - 1, current))
    return output


def infer_lattice(grid: np.ndarray) -> dict[str, Any]:
    """Rank likely cell scales/origins without assuming a grid game."""
    board = np.asarray(grid)
    run_lengths: Counter[int] = Counter()
    boundaries_y: Counter[int] = Counter()
    boundaries_x: Counter[int] = Counter()
    for row in board:
        runs = _runs(row)
        for start, end, _ in runs:
            run_lengths[end - start + 1] += 1
        for start, _, _ in runs[1:]:
            boundaries_x[start] += 1
    for col in board.T:
        runs = _runs(col)
        for start, end, _ in runs:
            run_lengths[end - start + 1] += 1
        for start, _, _ in runs[1:]:
            boundaries_y[start] += 1

    useful = [
        length
        for length, count in run_lengths.items()
        for _ in range(min(count, 20))
        if length > 1
    ]
    common_gcd = 0
    for length in useful:
        common_gcd = gcd(common_gcd, length)
    candidates: list[dict[str, Any]] = []
    for scale in range(2, min(17, max(board.shape))):
        support = sum(
            count for length, count in run_lengths.items() if length % scale == 0
        )
        if support == 0:
            continue
        origins = []
        for boundaries in (boundaries_y, boundaries_x):
            scores = Counter()
            for boundary, count in boundaries.items():
                scores[boundary % scale] += count
            origins.append(scores.most_common(1)[0][0] if scores else 0)
        candidates.append(
            {
                "scale": scale,
                "origin": origins,
                "support": support,
                "support_fraction": round(
                    support / max(1, sum(run_lengths.values())), 4
                ),
            }
        )
    candidates.sort(key=lambda item: (-item["support_fraction"], -item["scale"]))
    return {
        "common_run_gcd": common_gcd,
        "candidates": candidates[:6],
        "top_run_lengths": [
            {"length": length, "count": count}
            for length, count in run_lengths.most_common(12)
        ],
    }


def line_graph(grid: np.ndarray, *, min_length: int = 3) -> dict[str, Any]:
    """Describe long monochrome runs as a neutral track/line hypothesis."""
    board = np.asarray(grid)
    segments: list[dict[str, Any]] = []
    for row, values in enumerate(board):
        for start, end, color in _runs(values):
            if end - start + 1 >= min_length:
                segments.append(
                    {
                        "axis": "x",
                        "fixed": row,
                        "start": start,
                        "end": end,
                        "color": color,
                    }
                )
    for col, values in enumerate(board.T):
        for start, end, color in _runs(values):
            if end - start + 1 >= min_length:
                segments.append(
                    {
                        "axis": "y",
                        "fixed": col,
                        "start": start,
                        "end": end,
                        "color": color,
                    }
                )
    segments.sort(key=lambda item: (-(item["end"] - item["start"] + 1), item["color"]))
    return {"segments": segments[:160], "count": len(segments)}


def frame_delta(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(before), np.asarray(after)
    changed = np.argwhere(left != right)
    if not len(changed):
        return {"changed_cells": 0, "bbox": None, "color_changes": []}
    rows, cols = changed[:, 0], changed[:, 1]
    changes = Counter((int(left[y, x]), int(right[y, x])) for y, x in changed)
    return {
        "changed_cells": len(changed),
        "bbox": [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())],
        "color_changes": [
            {"from": old, "to": new, "count": count}
            for (old, new), count in changes.most_common(20)
        ],
        "cells": [
            {"y": int(y), "x": int(x), "from": int(left[y, x]), "to": int(right[y, x])}
            for y, x in changed[:200]
        ],
    }


def motion_trace(frames: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for index, (before, after) in enumerate(pairwise(frames)):
        prior = connected_components(before, min_size=2)
        current = connected_components(after, min_size=2)
        by_key: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for item in current:
            by_key[(item["color"], item["signature"], item["size"])].append(item)
        moved = []
        for item in prior:
            matches = by_key.get((item["color"], item["signature"], item["size"]), [])
            if len(matches) != 1:
                continue
            match = matches[0]
            if item["center"] != match["center"]:
                moved.append(
                    {
                        "color": item["color"],
                        "size": item["size"],
                        "signature": item["signature"],
                        "from": item["center"],
                        "to": match["center"],
                    }
                )
        output.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                "delta": frame_delta(before, after),
                "moved": moved[:24],
            }
        )
    return output


def transition_story(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Word a settled-frame transition as component-level events.

    Coordinates in lines are (x,y) with x=column and y=row, matching ACTION6.
    """
    left, right = np.asarray(before), np.asarray(after)
    empty: dict[str, Any] = {
        "moved": [],
        "vanished": [],
        "appeared": [],
        "recolored": [],
        "resized": [],
        "color_changes": [],
    }
    if left.shape != right.shape:
        return {
            **empty,
            "changed_cells": None,
            "bbox": None,
            "lines": [
                f"board size changed {left.shape[0]}x{left.shape[1]} → {right.shape[0]}x{right.shape[1]}"
            ],
        }
    delta = frame_delta(left, right)
    if not delta["changed_cells"]:
        return {
            **empty,
            "changed_cells": 0,
            "bbox": None,
            "lines": ["no visible change"],
        }
    box = delta["bbox"]
    lines = [
        f"{delta['changed_cells']} cells changed in rows {box[0]}..{box[1]}, cols {box[2]}..{box[3]}"
    ]
    story = {
        **empty,
        "changed_cells": delta["changed_cells"],
        "bbox": box,
        "color_changes": delta["color_changes"],
    }
    if delta["changed_cells"] > left.size * 3 // 5:
        flips = ", ".join(
            f"{item['from']:x}→{item['to']:x}×{item['count']}"
            for item in delta["color_changes"][:5]
        )
        story["lines"] = lines + [f"board largely redrawn; color flips {flips}"]
        return story

    huge = left.size // 2
    groups_before: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    groups_after: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in connected_components(left, min_size=1):
        if item["size"] <= huge:
            groups_before[(item["color"], item["signature"])].append(item)
    for item in connected_components(right, min_size=1):
        if item["size"] <= huge:
            groups_after[(item["color"], item["signature"])].append(item)

    moved: list[dict[str, Any]] = []
    gone: list[dict[str, Any]] = []
    born: list[dict[str, Any]] = []
    for key in set(groups_before) | set(groups_after):
        olds = groups_before.get(key, [])
        news = groups_after.get(key, [])
        new_positions = {tuple(item["bbox"]) for item in news}
        old_positions = {tuple(item["bbox"]) for item in olds}
        missing = [item for item in olds if tuple(item["bbox"]) not in new_positions]
        fresh = [item for item in news if tuple(item["bbox"]) not in old_positions]
        if len(missing) == 1 and len(fresh) == 1:
            old_item, new_item = missing[0], fresh[0]
            moved.append(
                {
                    "color": old_item["color"],
                    "size": old_item["size"],
                    "signature": old_item["signature"],
                    "from_bbox": old_item["bbox"],
                    "to_bbox": new_item["bbox"],
                    "from_center": old_item["center"],
                    "to_center": new_item["center"],
                    "dx": new_item["bbox"][2] - old_item["bbox"][2],
                    "dy": new_item["bbox"][0] - old_item["bbox"][0],
                }
            )
        else:
            gone.extend(missing)
            born.extend(fresh)

    recolored: list[dict[str, Any]] = []
    for old_item in list(gone):
        match = next(
            (
                item
                for item in born
                if item["signature"] == old_item["signature"]
                and item["bbox"] == old_item["bbox"]
            ),
            None,
        )
        if match is not None:
            recolored.append(
                {
                    "size": old_item["size"],
                    "bbox": old_item["bbox"],
                    "representative": old_item["representative"],
                    "from_color": old_item["color"],
                    "to_color": match["color"],
                }
            )
            gone.remove(old_item)
            born.remove(match)

    resized: list[dict[str, Any]] = []
    for old_item in list(gone):
        match = next(
            (
                item
                for item in born
                if item["color"] == old_item["color"]
                and not (
                    item["bbox"][1] < old_item["bbox"][0]
                    or old_item["bbox"][1] < item["bbox"][0]
                    or item["bbox"][3] < old_item["bbox"][2]
                    or old_item["bbox"][3] < item["bbox"][2]
                )
            ),
            None,
        )
        if match is not None:
            resized.append(
                {
                    "color": old_item["color"],
                    "from_size": old_item["size"],
                    "to_size": match["size"],
                    "representative": match["representative"],
                }
            )
            gone.remove(old_item)
            born.remove(match)

    def cap(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: -int(item.get("size", item.get("to_size", 0))))

    story.update(
        moved=cap(moved),
        vanished=cap(gone),
        appeared=cap(born),
        recolored=cap(recolored),
        resized=cap(resized),
    )
    limit = 6
    for item in story["moved"][:limit]:
        lines.append(
            f"color {item['color']:x} size {item['size']} moved (x,y) "
            f"({item['from_bbox'][2]},{item['from_bbox'][0]})→({item['to_bbox'][2]},{item['to_bbox'][0]}) "
            f"[dx={item['dx']:+d},dy={item['dy']:+d}]"
        )
    for item in story["recolored"][:limit]:
        y, x = item["representative"]
        lines.append(
            f"size {item['size']} at (x,y) ({x},{y}) recolored {item['from_color']:x}→{item['to_color']:x}"
        )
    for item in story["resized"][:limit]:
        y, x = item["representative"]
        lines.append(
            f"color {item['color']:x} resized {item['from_size']}→{item['to_size']} near (x,y) ({x},{y})"
        )
    for item in story["vanished"][:limit]:
        y, x = item["representative"]
        lines.append(
            f"color {item['color']:x} size {item['size']} vanished at (x,y) ({x},{y})"
        )
    for item in story["appeared"][:limit]:
        y, x = item["representative"]
        lines.append(
            f"color {item['color']:x} size {item['size']} appeared at (x,y) ({x},{y})"
        )
    for name in ("moved", "recolored", "resized", "vanished", "appeared"):
        extra = len(story[name]) - limit
        if extra > 0:
            lines.append(f"… {extra} more {name}")
    if len(lines) == 1:
        flips = ", ".join(
            f"{item['from']:x}→{item['to']:x}×{item['count']}"
            for item in delta["color_changes"][:5]
        )
        lines.append(f"color flips {flips}")
    story["lines"] = lines
    return story


def build_scene_dossier(
    paths: RunPaths,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cache a complete text/data scene read once per event."""
    tip = events[-1]
    settled = frame_at(tip)
    components = connected_components(settled, min_size=1)
    frames = []
    if len(events) > 1:
        frames.append(frame_at(events[-2]))
    frames.extend(frame_at(tip, index) for index in range(int(tip["n_frames"])))
    colors = Counter(int(value) for value in settled.reshape(-1))
    transition = motion_trace(frames)
    prior_actions = events[-2].get("available_actions", []) if len(events) > 1 else []
    record = {
        "version": 1,
        "event": int(tip["id"]),
        "level": int(tip["levels_completed"]),
        "win_levels": int(tip["win_levels"]),
        "state": tip["state"],
        "action": canonical_action(tip),
        "available_actions": list(tip.get("available_actions", [])),
        "action_set_delta": {
            "added": sorted(set(tip.get("available_actions", [])) - set(prior_actions)),
            "removed": sorted(
                set(prior_actions) - set(tip.get("available_actions", []))
            ),
        },
        "scene": {
            "shape": list(settled.shape),
            "colors": [
                {"color": color, "cells": count}
                for color, count in colors.most_common()
            ],
            "components": [
                {key: value for key, value in item.items() if key != "shape_key"}
                for item in components[:320]
            ],
            "component_count": len(components),
            "repeated_shapes": repeated_shapes(settled),
            "lattice": infer_lattice(settled),
            "lines": line_graph(settled),
        },
        "transition": {
            "frame_count": int(tip["n_frames"]),
            "settled": frame_delta(frame_at(events[-2]), settled)
            if len(events) > 1
            else None,
            "frames": transition,
        },
    }
    atomic_json(paths.dossier, record)
    return record
