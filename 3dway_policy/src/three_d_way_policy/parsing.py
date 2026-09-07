"""Safe parsing for NVILA trajectory responses."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_POINT_OR_ACTION = re.compile(
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"
    r"|<action>\s*(Open|Close) Gripper\s*</action>",
    flags=re.IGNORECASE,
)
_VIEW_TAG = re.compile(
    r"<ans_view(?P<view>[12])>(?P<body>.*?)</ans_view(?P=view)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _validate_pair(trajectories: Sequence[NDArray[np.float64]]) -> None:
    if len(trajectories) != 2:
        raise ValueError(f"Expected two trajectories, got {len(trajectories)}.")
    if trajectories[0].shape != trajectories[1].shape:
        raise ValueError(
            "View trajectories must have matching shapes, got "
            f"{trajectories[0].shape} and {trajectories[1].shape}."
        )
    if not np.array_equal(trajectories[0][:, 2], trajectories[1][:, 2]):
        raise ValueError("Gripper actions do not match between view1 and view2.")


def _parse_tagged_body(body: str) -> NDArray[np.float64]:
    points: list[list[float]] = []
    current_action = GRIPPER_CLOSE

    for match in _POINT_OR_ACTION.finditer(body):
        x, y, action = match.groups()
        if action is None:
            points.append([float(x), float(y), float(current_action)])
            continue
        if not points:
            raise ValueError("A gripper action must follow a trajectory point.")
        current_action = GRIPPER_OPEN if action.lower() == "open" else GRIPPER_CLOSE
        points[-1][2] = float(current_action)

    if not points:
        raise ValueError("No trajectory points were found in a tagged view.")
    result = np.asarray(points, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Trajectory contains a non-finite value.")
    return result


def parse_response(text: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse the `<ans_view1>`/`<ans_view2>` response format."""

    matches = {match.group("view"): match.group("body") for match in _VIEW_TAG.finditer(text)}
    if set(matches) != {"1", "2"}:
        raise ValueError("Response must contain exactly ans_view1 and ans_view2 tags.")

    trajectories = (_parse_tagged_body(matches["1"]), _parse_tagged_body(matches["2"]))
    _validate_pair(trajectories)
    return trajectories
