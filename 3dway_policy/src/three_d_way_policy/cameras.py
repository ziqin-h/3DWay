"""Camera models and calibration loading."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Camera:
    """A calibrated pinhole camera using a camera-to-world transform."""

    name: str
    width: int
    height: int
    intrinsics: NDArray[np.float64]
    camera_to_world: NDArray[np.float64]
    source_camera: str | None = None

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world, dtype=np.float64)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}.")
        if camera_to_world.shape != (4, 4):
            raise ValueError(
                f"camera_to_world must have shape (4, 4), got {camera_to_world.shape}."
            )
        if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
            raise ValueError("Camera matrices must contain only finite values.")
        if np.isclose(np.linalg.det(intrinsics), 0.0):
            raise ValueError("Camera intrinsics matrix is singular.")
        if not np.allclose(camera_to_world[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("camera_to_world must be a homogeneous transform.")

        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "Camera":
        return cls(
            name=name,
            source_camera=value.get("source_camera"),
            width=int(value["width"]),
            height=int(value["height"]),
            intrinsics=value["intrinsics"],
            camera_to_world=value["camera_to_world"],
        )

    @property
    def world_to_camera(self) -> NDArray[np.float64]:
        return np.linalg.inv(self.camera_to_world)

    def rays_from_normalized_points(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return world-frame ray origins and unit directions for Nx2 points."""

        normalized = np.asarray(points, dtype=np.float64)
        if normalized.ndim != 2 or normalized.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {normalized.shape}.")
        if not np.isfinite(normalized).all():
            raise ValueError("Trajectory points must contain only finite values.")

        pixels = np.column_stack(
            (
                normalized[:, 0] * self.width,
                normalized[:, 1] * self.height,
                np.ones(len(normalized), dtype=np.float64),
            )
        )
        camera_directions = np.linalg.solve(self.intrinsics, pixels.T).T
        world_directions = camera_directions @ self.camera_to_world[:3, :3].T
        norms = np.linalg.norm(world_directions, axis=1, keepdims=True)
        if np.any(np.isclose(norms, 0.0)):
            raise ValueError("A trajectory point produced a zero-length camera ray.")

        world_directions /= norms
        origins = np.repeat(self.camera_to_world[None, :3, 3], len(normalized), axis=0)
        return origins, world_directions

    def project_world_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Project Nx3 world points to normalized image coordinates."""

        world_points = np.asarray(points, dtype=np.float64)
        if world_points.ndim != 2 or world_points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {world_points.shape}.")

        homogeneous = np.column_stack((world_points, np.ones(len(world_points))))
        camera_points = (self.world_to_camera @ homogeneous.T).T[:, :3]
        if np.any(np.isclose(camera_points[:, 2], 0.0)):
            raise ValueError("Cannot project a point on the camera plane.")

        pixels_h = camera_points @ self.intrinsics.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        return pixels / np.array([self.width, self.height], dtype=np.float64)


def load_camera_rig(path: str | Path) -> "OrderedDict[str, Camera]":
    """Load an ordered camera rig from the project JSON schema."""

    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)

    if document.get("schema_version") != 1:
        raise ValueError("Unsupported camera calibration schema version.")
    if document.get("extrinsics_convention") != "camera_to_world":
        raise ValueError("Only camera_to_world extrinsics are supported.")

    views = document.get("views")
    if not isinstance(views, dict) or len(views) < 2:
        raise ValueError("Calibration must contain at least two camera views.")

    return OrderedDict((name, Camera.from_mapping(name, value)) for name, value in views.items())

