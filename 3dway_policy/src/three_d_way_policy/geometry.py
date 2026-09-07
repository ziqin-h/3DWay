"""Multi-view trajectory geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cameras import Camera


@dataclass(frozen=True)
class TriangulationResult:
    """Triangulated points and the residual between corresponding rays."""

    points: NDArray[np.float64]
    ray_residuals: NDArray[np.float64]


def closest_points_between_rays(
    origin_1: ArrayLike,
    direction_1: ArrayLike,
    origin_2: ArrayLike,
    direction_2: ArrayLike,
    *,
    constrain_to_rays: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find the closest point on each of two 3D rays."""

    p1 = np.asarray(origin_1, dtype=np.float64)
    p2 = np.asarray(origin_2, dtype=np.float64)
    v1 = np.asarray(direction_1, dtype=np.float64)
    v2 = np.asarray(direction_2, dtype=np.float64)
    if any(value.shape != (3,) for value in (p1, p2, v1, v2)):
        raise ValueError("Ray origins and directions must have shape (3,).")

    delta = p2 - p1
    coefficients = np.array(
        [
            [np.dot(v1, v1), -np.dot(v1, v2)],
            [-np.dot(v1, v2), np.dot(v2, v2)],
        ],
        dtype=np.float64,
    )
    right_hand_side = np.array(
        [np.dot(v1, delta), -np.dot(v2, delta)], dtype=np.float64
    )

    parameters, _, _, _ = np.linalg.lstsq(coefficients, right_hand_side, rcond=None)
    if constrain_to_rays:
        parameters = np.maximum(parameters, 0.0)

    return p1 + parameters[0] * v1, p2 + parameters[1] * v2


def triangulate_trajectories(
    cameras: Sequence[Camera], trajectories: Sequence[ArrayLike]
) -> TriangulationResult:
    """Triangulate corresponding normalized 2D points from exactly two views."""

    if len(cameras) != 2 or len(trajectories) != 2:
        raise ValueError("Exactly two cameras and two trajectories are required.")

    points_1 = np.asarray(trajectories[0], dtype=np.float64)
    points_2 = np.asarray(trajectories[1], dtype=np.float64)
    if points_1.ndim != 2 or points_1.shape[1] != 2:
        raise ValueError(f"First trajectory must have shape (N, 2), got {points_1.shape}.")
    if points_2.shape != points_1.shape:
        raise ValueError(
            f"Trajectory shapes must match, got {points_1.shape} and {points_2.shape}."
        )
    if len(points_1) == 0:
        raise ValueError("Trajectories must contain at least one point.")

    origins_1, directions_1 = cameras[0].rays_from_normalized_points(points_1)
    origins_2, directions_2 = cameras[1].rays_from_normalized_points(points_2)

    triangulated = []
    residuals = []
    for p1, v1, p2, v2 in zip(origins_1, directions_1, origins_2, directions_2):
        endpoint_1, endpoint_2 = closest_points_between_rays(p1, v1, p2, v2)
        triangulated.append((endpoint_1 + endpoint_2) / 2.0)
        residuals.append(np.linalg.norm(endpoint_1 - endpoint_2))

    return TriangulationResult(
        points=np.asarray(triangulated, dtype=np.float64),
        ray_residuals=np.asarray(residuals, dtype=np.float64),
    )


def to_robot_actions(
    points: ArrayLike,
    gripper_states: ArrayLike,
    *,
    orientation_xyzw: ArrayLike = (0.0, 1.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Combine XYZ, a fixed XYZW quaternion, and gripper state into Nx8 actions."""

    positions = np.asarray(points, dtype=np.float64)
    gripper = np.asarray(gripper_states, dtype=np.float64)
    orientation = np.asarray(orientation_xyzw, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {positions.shape}.")
    if gripper.shape != (len(positions),):
        raise ValueError(
            f"gripper_states must have shape ({len(positions)},), got {gripper.shape}."
        )
    if orientation.shape != (4,):
        raise ValueError(f"orientation_xyzw must have shape (4,), got {orientation.shape}.")
    if not np.isclose(np.linalg.norm(orientation), 1.0):
        raise ValueError("orientation_xyzw must be a unit quaternion.")

    orientations = np.repeat(orientation[None, :], len(positions), axis=0)
    return np.column_stack((positions, orientations, gripper))

