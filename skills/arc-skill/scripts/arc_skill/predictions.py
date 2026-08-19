"""A tiny prediction vocabulary the runtime grades automatically.

Every live action carries a prediction. Structured claims are graded against
the settled result; free text is graded as "some visible change". Coordinates
are x=column, y=row — the same convention as ACTION6.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import ArcSkillError, frame_at

CLAIMS_HELP = """\
PREDICTION CLAIMS | separate several with ";" | x=column y=row (like ACTION6)
  noop                 nothing visible changes
  change               something visible changes (free text means this too)
  cell X,Y=V           cell at column X, row Y becomes hex color V
  move X,Y DX,DY       the object covering X,Y shifts by (DX,DY) and vacates its old cells
  vanish X,Y           every cell of the object covering X,Y stops being its color
  region X0:X1,Y0:Y1   all changes fall inside this half-open box, and something changes
  level+1              this action completes the level
  win                  this action wins the game
Free text that is not a claim is kept as commentary. Example:
  --predict "door opens; cell 12,5=0; region 10:14,3:8"
"""

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("noop", re.compile(r"^noop$", re.IGNORECASE)),
    ("change", re.compile(r"^change$", re.IGNORECASE)),
    ("win", re.compile(r"^win$", re.IGNORECASE)),
    ("level_up", re.compile(r"^level\s*\+\s*1$", re.IGNORECASE)),
    (
        "cell",
        re.compile(r"^cell\s+(\d+)\s*,\s*(\d+)\s*=\s*([0-9a-fA-F])$", re.IGNORECASE),
    ),
    (
        "move",
        re.compile(
            r"^move\s+(\d+)\s*,\s*(\d+)\s+([+-]?\d+)\s*,\s*([+-]?\d+)$", re.IGNORECASE
        ),
    ),
    ("vanish", re.compile(r"^vanish\s+(\d+)\s*,\s*(\d+)$", re.IGNORECASE)),
    (
        "region",
        re.compile(r"^region\s+(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)$", re.IGNORECASE),
    ),
)

_KEYWORD = re.compile(
    r"^(noop|change|cell|move|vanish|region|level|win)\b", re.IGNORECASE
)


def parse_claims(text: str) -> list[dict[str, Any]]:
    """Parse a prediction string into claims; free text implies `change`."""
    if not text or not text.strip():
        raise ArcSkillError(
            f"an empty prediction predicts nothing; say what you expect\n{CLAIMS_HELP}"
        )
    claims: list[dict[str, Any]] = []
    for raw in text.split(";"):
        part = raw.strip()
        if not part:
            continue
        matched = False
        for kind, pattern in _PATTERNS:
            found = pattern.match(part)
            if not found:
                continue
            claim: dict[str, Any] = {"kind": kind, "text": part}
            if kind == "cell":
                claim.update(
                    x=int(found.group(1)),
                    y=int(found.group(2)),
                    value=int(found.group(3), 16),
                )
            elif kind == "move":
                claim.update(
                    x=int(found.group(1)),
                    y=int(found.group(2)),
                    dx=int(found.group(3)),
                    dy=int(found.group(4)),
                )
            elif kind == "vanish":
                claim.update(x=int(found.group(1)), y=int(found.group(2)))
            elif kind == "region":
                claim.update(
                    x0=int(found.group(1)),
                    x1=int(found.group(2)),
                    y0=int(found.group(3)),
                    y1=int(found.group(4)),
                )
            claims.append(claim)
            matched = True
            break
        if not matched:
            if _KEYWORD.match(part):
                raise ArcSkillError(f"malformed claim {part!r}\n{CLAIMS_HELP}")
            claims.append({"kind": "note", "text": part})
    gradable = [claim for claim in claims if claim["kind"] != "note"]
    if not gradable:
        # A prose prediction still commits to a visible effect.
        claims.append({"kind": "change", "text": "change (implied by free text)"})
    return claims


def _flood(grid: np.ndarray, x: int, y: int) -> tuple[int, set[tuple[int, int]]]:
    color = int(grid[y, x])
    height, width = grid.shape
    stack = [(y, x)]
    cells = {(y, x)}
    while stack:
        row, col = stack.pop()
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (
                0 <= nr < height
                and 0 <= nc < width
                and (nr, nc) not in cells
                and int(grid[nr, nc]) == color
            ):
                cells.add((nr, nc))
                stack.append((nr, nc))
    return color, cells


def grade_claims(
    claims: Sequence[Mapping[str, Any]],
    before: np.ndarray,
    event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Grade each claim against the settled result; ok plus a short actual."""
    after = frame_at(event)
    same_shape = before.shape == after.shape
    changed = int(np.count_nonzero(before != after)) if same_shape else None
    level_before = event.get("level_before")
    level_advanced = (
        level_before is not None
        and int(event["levels_completed"]) > int(level_before)
    )
    graded: list[dict[str, Any]] = []
    for claim in claims:
        kind = claim["kind"]
        if kind == "note":
            continue
        ok = False
        actual = ""
        if kind == "noop":
            ok = same_shape and changed == 0
            actual = (
                "no visible change"
                if ok
                else f"{'board size changed' if not same_shape else f'{changed} cells changed'}"
            )
        elif kind == "change":
            ok = (not same_shape) or bool(changed) or level_advanced
            actual = (
                "no visible change (0 cells)"
                if not ok
                else f"{changed if same_shape else 'board'} changed"
            )
        elif kind == "win":
            ok = str(event["state"]) == "WIN"
            actual = f"state {event['state']}"
        elif kind == "level_up":
            ok = level_advanced
            actual = (
                f"level advanced to {int(event['levels_completed'])} completed"
                if level_advanced
                else "level did not advance"
            )
        elif kind == "cell":
            x, y = int(claim["x"]), int(claim["y"])
            if not (same_shape and 0 <= y < after.shape[0] and 0 <= x < after.shape[1]):
                actual = "coordinates out of bounds"
            else:
                ok = int(after[y, x]) == int(claim["value"])
                actual = f"cell ({x},{y}) is {int(after[y, x]):x}"
        elif kind == "vanish":
            x, y = int(claim["x"]), int(claim["y"])
            if not (same_shape and 0 <= y < before.shape[0] and 0 <= x < before.shape[1]):
                actual = "coordinates out of bounds"
            else:
                color, cells = _flood(before, x, y)
                remaining = sum(
                    1 for row, col in cells if int(after[row, col]) == color
                )
                ok = remaining == 0
                actual = (
                    f"object of color {color:x} gone from its cells"
                    if ok
                    else f"{remaining}/{len(cells)} cells still color {color:x}"
                )
        elif kind == "move":
            x, y = int(claim["x"]), int(claim["y"])
            dx, dy = int(claim["dx"]), int(claim["dy"])
            if not (same_shape and 0 <= y < before.shape[0] and 0 <= x < before.shape[1]):
                actual = "coordinates out of bounds"
            else:
                color, cells = _flood(before, x, y)
                shifted = {(row + dy, col + dx) for row, col in cells}
                in_bounds = all(
                    0 <= row < after.shape[0] and 0 <= col < after.shape[1]
                    for row, col in shifted
                )
                if not in_bounds:
                    actual = "shifted object would leave the board"
                else:
                    arrived = all(
                        int(after[row, col]) == color for row, col in shifted
                    )
                    vacated = all(
                        int(after[row, col]) != color
                        for row, col in cells - shifted
                    )
                    ok = arrived and vacated
                    if ok:
                        actual = f"object moved by ({dx:+d},{dy:+d})"
                    elif not arrived:
                        actual = f"target cells are not color {color:x}"
                    else:
                        actual = f"old cells still color {color:x} (copied, not moved)"
        elif kind == "region":
            x0, x1 = int(claim["x0"]), int(claim["x1"])
            y0, y1 = int(claim["y0"]), int(claim["y1"])
            if not same_shape:
                actual = "board size changed"
            elif not changed:
                actual = "no visible change"
            else:
                outside = [
                    (int(col), int(row))
                    for row, col in np.argwhere(before != after)
                    if not (x0 <= col < x1 and y0 <= row < y1)
                ]
                ok = not outside
                actual = (
                    f"all {changed} changes inside the box"
                    if ok
                    else f"{len(outside)} changes outside, e.g. (x,y) {outside[0]}"
                )
        graded.append({**dict(claim), "ok": bool(ok), "actual": actual})
    return graded


def grade_lines(graded: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{'✓' if item['ok'] else '✗'} {item['text']}"
        + ("" if item["ok"] else f" — {item['actual']}")
        for item in graded
    ]
