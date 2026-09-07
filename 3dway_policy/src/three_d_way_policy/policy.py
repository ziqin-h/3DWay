"""NVILA-backed multi-view policy."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .cameras import Camera
from .geometry import to_robot_actions, triangulate_trajectories
from .parsing import parse_response


PROMPT = """You are given two views of the same task:
1. View 1: Execute the instruction in <quest>{instruction}</quest>.
2. View 2: Provide the corresponding trajectory for the same motion.

Return exactly two tagged lists:
<ans_view1>[(x, y), <action>Open Gripper</action>, ...]</ans_view1>
<ans_view2>[(x, y), <action>Open Gripper</action>, ...]</ans_view2>

Each (x, y) is a normalized image coordinate. Use only the actions
<action>Open Gripper</action> and <action>Close Gripper</action>.
The views must contain the same number of points and matching gripper actions.
"""

@dataclass(frozen=True)
class PolicyPrediction:
    """All intermediate and final outputs produced by the policy."""

    response_text: str
    trajectories_2d: tuple[NDArray[np.float64], NDArray[np.float64]]
    trajectory_3d: NDArray[np.float64]
    ray_residuals: NDArray[np.float64]
    actions: NDArray[np.float64]
    inference_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "trajectories_2d": [trajectory.tolist() for trajectory in self.trajectories_2d],
            "trajectory_3d": self.trajectory_3d.tolist(),
            "ray_residuals": self.ray_residuals.tolist(),
            "actions": self.actions.tolist(),
            "inference_seconds": self.inference_seconds,
        }


class NVILAPolicy:
    """Query an OpenAI-compatible NVILA server and triangulate its trajectories."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8888",
        api_key: str = "not-used",
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI # type: ignore[import]
            except ImportError as error:
                raise RuntimeError(
                    "The openai package is required for online inference. "
                    "Install the 3dway-policy package first."
                ) from error
            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    @staticmethod
    def _encode_image(image: Image.Image | NDArray[np.generic]) -> str:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(f"Image array must have shape (H, W, 3/4), got {array.shape}.")
            if array.dtype != np.uint8:
                raise ValueError("Image arrays must use uint8 RGB/RGBA values.")
            pil_image = Image.fromarray(array).convert("RGB")

        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _response_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip()
            if hasattr(first, "text"):
                return str(first.text).strip()
        raise ValueError("NVILA returned an unsupported message content structure.")

    def request_trajectories(
        self,
        images: Sequence[Image.Image | NDArray[np.generic]],
        instruction: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        use_cache: bool = True,
    ) -> tuple[str, float]:
        """Send two images and an instruction to the configured NVILA server."""

        if len(images) != 2:
            raise ValueError(f"Exactly two images are required, got {len(images)}.")
        if not instruction.strip():
            raise ValueError("instruction must not be empty.")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image)}"},
            }
            for image in images
        ]
        content.append(
            {
                "type": "text",
                "text": PROMPT.format(instruction=instruction.strip()),
            }
        )

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={"num_beams": num_beams, "use_cache": use_cache},
        )
        elapsed = time.perf_counter() - started
        return self._response_text(response.choices[0].message.content), elapsed

    def prediction_from_response(
        self,
        response_text: str,
        cameras: Sequence[Camera],
        *,
        inference_seconds: float | None = None,
    ) -> PolicyPrediction:
        """Parse and triangulate an existing model response."""

        camera_list = list(cameras)
        if len(camera_list) != 2:
            raise ValueError(f"Exactly two cameras are required, got {len(camera_list)}.")

        trajectories = parse_response(response_text)
        triangulation = triangulate_trajectories(
            camera_list, (trajectories[0][:, :2], trajectories[1][:, :2])
        )
        actions = to_robot_actions(triangulation.points, trajectories[0][:, 2])
        return PolicyPrediction(
            response_text=response_text,
            trajectories_2d=trajectories,
            trajectory_3d=triangulation.points,
            ray_residuals=triangulation.ray_residuals,
            actions=actions,
            inference_seconds=inference_seconds,
        )

    def predict(
        self,
        images: Sequence[Image.Image | NDArray[np.generic]],
        cameras: Sequence[Camera],
        instruction: str,
        **generation_options: Any,
    ) -> PolicyPrediction:
        """Run online inference and convert the result into robot actions."""

        response_text, elapsed = self.request_trajectories(
            images, instruction, **generation_options
        )
        return self.prediction_from_response(
            response_text, cameras, inference_seconds=elapsed
        )

