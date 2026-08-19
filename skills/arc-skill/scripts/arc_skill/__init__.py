from dataclasses import dataclass


@dataclass(frozen=True)
class Unknown:
    """An explicitly unmodeled transition, not a failed prediction."""

    reason: str = "unspecified transition"


def connected_components(*args, **kwargs):
    from .perception import connected_components as implementation

    return implementation(*args, **kwargs)


def frame_delta(*args, **kwargs):
    from .perception import frame_delta as implementation

    return implementation(*args, **kwargs)


def infer_lattice(*args, **kwargs):
    from .perception import infer_lattice as implementation

    return implementation(*args, **kwargs)


def line_graph(*args, **kwargs):
    from .perception import line_graph as implementation

    return implementation(*args, **kwargs)


def motion_trace(*args, **kwargs):
    from .perception import motion_trace as implementation

    return implementation(*args, **kwargs)


def repeated_shapes(*args, **kwargs):
    from .perception import repeated_shapes as implementation

    return implementation(*args, **kwargs)


def transition_story(*args, **kwargs):
    from .perception import transition_story as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "Unknown",
    "connected_components",
    "frame_delta",
    "infer_lattice",
    "line_graph",
    "motion_trace",
    "repeated_shapes",
    "transition_story",
]
