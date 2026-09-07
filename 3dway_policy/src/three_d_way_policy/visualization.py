"""Save 2D policy artifacts and visualize saved 3D predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw

from .parsing import GRIPPER_CLOSE, GRIPPER_OPEN
from .policy import PolicyPrediction


@dataclass(frozen=True)
class OutputArtifacts:
    """Paths written for one policy prediction."""

    result_json: Path
    view_images: tuple[Path, Path]


def _jet_colors(count: int) -> NDArray[np.uint8]:
    """Return blue-to-red colors matching the classic ``jet`` color map."""

    if count < 1:
        return np.empty((0, 3), dtype=np.uint8)
    positions = np.linspace(0.0, 1.0, count)
    red = np.clip(1.5 - np.abs(4.0 * positions - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * positions - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * positions - 1.0), 0.0, 1.0)
    return np.rint(np.column_stack((red, green, blue)) * 255.0).astype(np.uint8)


def _interpolate_polyline(points: NDArray[np.float64], samples: int) -> NDArray[np.float64]:
    if len(points) < 2:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if np.isclose(distances[-1], 0.0):
        return np.repeat(points[:1], max(samples, 1), axis=0)

    sample_distances = np.linspace(0.0, distances[-1], max(samples, 2))
    return np.column_stack(
        [np.interp(sample_distances, distances, points[:, axis]) for axis in range(2)]
    )


def draw_trajectory_2d(
    image: Image.Image | NDArray[np.generic],
    trajectory: ArrayLike,
    *,
    line_width: int = 3,
    marker_radius: int = 7,
    interpolation_samples: int = 100,
) -> Image.Image:
    """Draw a rainbow trajectory and gripper-change markers on an RGB image."""

    if isinstance(image, Image.Image):
        output = image.convert("RGB").copy()
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Image must have shape (H, W, 3/4), got {array.shape}.")
        if array.dtype != np.uint8:
            raise ValueError("Image arrays must use uint8 values.")
        output = Image.fromarray(array).convert("RGB")

    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"trajectory must have shape (N, 3), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("trajectory must contain only finite values.")

    width, height = output.size
    pixels = points[:, :2] * np.array([width, height], dtype=np.float64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    interpolated = _interpolate_polyline(pixels, interpolation_samples)

    draw = ImageDraw.Draw(output)
    colors = _jet_colors(max(len(interpolated) - 1, 0))
    for index in range(len(interpolated) - 1):
        start = tuple(np.rint(interpolated[index]).astype(int))
        end = tuple(np.rint(interpolated[index + 1]).astype(int))
        draw.line((start, end), fill=tuple(colors[index]), width=line_width)

    gripper_states = points[:, 2].astype(int)
    for index, (x, y) in enumerate(pixels):
        if index > 0 and gripper_states[index] == gripper_states[index - 1]:
            continue
        fill = (40, 110, 255) if gripper_states[index] == GRIPPER_OPEN else (230, 45, 45)
        center_x, center_y = int(round(x)), int(round(y))
        bounds = (
            center_x - marker_radius,
            center_y - marker_radius,
            center_x + marker_radius,
            center_y + marker_radius,
        )
        draw.ellipse(bounds, fill=fill, outline=(255, 255, 255), width=2)

    return output


def save_prediction_outputs(
    prediction: PolicyPrediction,
    images: Sequence[Image.Image | NDArray[np.generic]],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> OutputArtifacts:
    """Save prediction JSON and paired 2D trajectory visualizations."""

    if len(images) != 2:
        raise ValueError(f"Exactly two images are required, got {len(images)}.")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    image_paths = (directory / "view1_trajectory.png", directory / "view2_trajectory.png")

    for image, trajectory, path in zip(images, prediction.trajectories_2d, image_paths):
        draw_trajectory_2d(image, trajectory).save(path)

    result_path = directory / "prediction.json"
    document = prediction.as_dict()
    document["metadata"] = dict(metadata or {})
    document["artifacts"] = {
        "view1_trajectory": image_paths[0].name,
        "view2_trajectory": image_paths[1].name,
    }
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    return OutputArtifacts(result_json=result_path, view_images=image_paths)


def load_prediction_result(
    result_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load the 3D trajectory and gripper states from a saved prediction JSON."""

    path = Path(result_path)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    trajectory = np.asarray(document.get("trajectory_3d"), dtype=np.float64)
    actions = np.asarray(document.get("actions"), dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) == 0:
        raise ValueError(f"Invalid trajectory_3d in {path}.")
    if actions.ndim != 2 or actions.shape != (len(trajectory), 8):
        raise ValueError(f"Invalid actions in {path}; expected shape ({len(trajectory)}, 8).")
    if not np.isfinite(trajectory).all() or not np.isfinite(actions).all():
        raise ValueError(f"Prediction contains non-finite values: {path}.")

    gripper_states = actions[:, -1].astype(np.int64)
    if not np.isin(gripper_states, [GRIPPER_CLOSE, GRIPPER_OPEN]).all():
        raise ValueError(f"Prediction contains invalid gripper states: {path}.")
    return trajectory, gripper_states


def visualize_prediction_3d(
    scene_path: str | Path,
    result_path: str | Path,
    *,
    marker_radius: float = 0.015,
    window_name: str = "3DWay prediction",
) -> None:
    """Overlay a saved 3D trajectory on a PLY scene in an Open3D window."""

    try:
        import open3d as o3d # type: ignore[import]
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for 3D visualization. Install it with "
            "`python -m pip install -e './3dway_policy[visualization]'`."
        ) from error

    if marker_radius <= 0:
        raise ValueError("marker_radius must be positive.")

    scene = o3d.io.read_point_cloud(str(scene_path))
    if scene.is_empty():
        raise ValueError(f"Scene point cloud is empty or unreadable: {scene_path}.")

    trajectory, gripper_states = load_prediction_result(result_path)
    geometries: list[Any] = [scene]

    trajectory_cloud = o3d.geometry.PointCloud()
    trajectory_cloud.points = o3d.utility.Vector3dVector(trajectory)
    trajectory_cloud.colors = o3d.utility.Vector3dVector(_jet_colors(len(trajectory)) / 255.0)
    geometries.append(trajectory_cloud)

    if len(trajectory) > 1:
        lines = np.column_stack((np.arange(len(trajectory) - 1), np.arange(1, len(trajectory))))
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(
            _jet_colors(len(trajectory) - 1) / 255.0
        )
        geometries.append(line_set)

    for index, (point, state) in enumerate(zip(trajectory, gripper_states)):
        if index > 0 and state == gripper_states[index - 1]:
            continue
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        marker.translate(point)
        marker.paint_uniform_color(
            [0.16, 0.43, 1.0] if state == GRIPPER_OPEN else [0.90, 0.18, 0.18]
        )
        marker.compute_vertex_normals()
        geometries.append(marker)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720,
    )
