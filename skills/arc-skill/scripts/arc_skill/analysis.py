from __future__ import annotations

import ast
import heapq
import math
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .core import ArcSkillError, RunPaths, canonical_action, frame_at, load_events
from .perception import (
    connected_components,
    frame_delta,
    infer_lattice,
    line_graph,
    motion_trace,
    repeated_shapes,
)


def show(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim != 2:
        return repr(value)
    return "\n".join("".join(format(int(cell), "x") for cell in row) for row in array)


def crop(grid: np.ndarray, rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
    return np.asarray(grid)[rows[0] : rows[1], cols[0] : cols[1]].copy()


def sample(grid: np.ndarray, coordinates: Iterable[tuple[int, int]]) -> list[int]:
    array = np.asarray(grid)
    return [int(array[row, column]) for row, column in coordinates]


def _neighbors(
    cell: tuple[int, int], diagonal: bool = False
) -> tuple[tuple[int, int], ...]:
    row, column = cell
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return tuple((row + dr, column + dc) for dr, dc in steps)


def shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: np.ndarray | Callable[[tuple[int, int]], bool],
    *,
    diagonal: bool = False,
) -> list[tuple[int, int]] | None:
    """BFS over a caller-supplied passability mask. Coordinates are (row, col)."""
    if callable(passable):
        allowed = passable
    else:
        mask = np.asarray(passable, dtype=bool)
        allowed = lambda cell: (
            0 <= cell[0] < mask.shape[0]
            and 0 <= cell[1] < mask.shape[1]
            and bool(mask[cell])
        )
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            route: list[tuple[int, int]] = []
            cursor: tuple[int, int] | None = current
            while cursor is not None:
                route.append(cursor)
                cursor = parent[cursor]
            return route[::-1]
        for successor in _neighbors(current, diagonal):
            if successor not in parent and allowed(successor):
                parent[successor] = current
                queue.append(successor)
    return None


def bfs(
    start: Any,
    is_goal: Callable[[Any], bool],
    expand: Callable[[Any], Iterable[tuple[Any, Any]]],
    *,
    key: Callable[[Any], Any] = repr,
    max_nodes: int = 100_000,
) -> tuple[list[Any] | None, int]:
    queue = deque([(start, ())])
    seen = {key(start)}
    nodes = 0
    while queue and nodes < max_nodes:
        state, route = queue.popleft()
        nodes += 1
        if is_goal(state):
            return list(route), nodes
        for action, successor in expand(state):
            marker = key(successor)
            if marker not in seen:
                seen.add(marker)
                queue.append((successor, route + (action,)))
    return None, nodes


def astar(
    start: Any,
    is_goal: Callable[[Any], bool],
    expand: Callable[[Any], Iterable[tuple[Any, Any, float]]],
    heuristic: Callable[[Any], float],
    *,
    key: Callable[[Any], Any] = repr,
    max_nodes: int = 100_000,
) -> tuple[list[Any] | None, int]:
    queue: list[tuple[float, int, float, Any, tuple[Any, ...]]] = [
        (float(heuristic(start)), 0, 0.0, start, ())
    ]
    best = {key(start): 0.0}
    sequence = nodes = 0
    while queue and nodes < max_nodes:
        _, _, cost, state, route = heapq.heappop(queue)
        if cost != best.get(key(state)):
            continue
        nodes += 1
        if is_goal(state):
            return list(route), nodes
        for action, successor, edge_cost in expand(state):
            next_cost = cost + float(edge_cost)
            marker = key(successor)
            if next_cost >= best.get(marker, math.inf):
                continue
            best[marker] = next_cost
            sequence += 1
            heapq.heappush(
                queue,
                (
                    next_cost + float(heuristic(successor)),
                    sequence,
                    next_cost,
                    successor,
                    route + (action,),
                ),
            )
    return None, nodes


def namespace(paths: RunPaths) -> dict[str, Any]:
    events = load_events(paths)
    settled = [frame_at(event) for event in events]
    frames = [
        [frame_at(event, index) for index in range(len(event["frames"]))]
        for event in events
    ]
    transitions = [
        {
            "event": int(event["id"]),
            "action": canonical_action(event),
            "before": settled[index - 1],
            "after": settled[index],
            "frames": frames[index],
            "state": event["state"],
            "level": int(event["levels_completed"]) + 1,
        }
        for index, event in enumerate(events)
        if index
    ]
    return {
        "np": np,
        "grid": settled[-1],
        "previous": settled[-2] if len(settled) > 1 else None,
        "settled": settled,
        "frames": frames,
        "events": events,
        "transitions": transitions,
        "actions": [canonical_action(event) for event in events[1:]],
        "show": show,
        "crop": crop,
        "sample": sample,
        "connected_components": connected_components,
        "frame_delta": frame_delta,
        "infer_lattice": infer_lattice,
        "line_graph": line_graph,
        "motion_trace": motion_trace,
        "repeated_shapes": repeated_shapes,
        "shortest_path": shortest_path,
        "bfs": bfs,
        "astar": astar,
    }


def run_python(paths: RunPaths, source: str) -> Any:
    scope = namespace(paths)
    try:
        tree = ast.parse(source, mode="exec")
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            value = eval(
                compile(ast.Expression(tree.body[0].value), "<arc-python>", "eval"),
                scope,
                scope,
            )
            if value is not None:
                print(
                    show(value)
                    if isinstance(value, np.ndarray) and value.ndim == 2
                    else repr(value)
                )
            return value
        exec(compile(tree, "<arc-python>", "exec"), scope, scope)  # noqa: S102 - deliberate local analysis console
        return None
    except Exception as error:
        raise ArcSkillError(
            f"analysis failed: {type(error).__name__}: {error}"
        ) from error
