#!/usr/bin/env python3
"""Run the 3DWay policy on the checked-in paired-view example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from three_d_way_policy import (
    NVILAPolicy,
    load_camera_rig,
    save_prediction_outputs,
    visualize_prediction_3d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Checkpoint directory name expected by the NVILA server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--instruction", default="put the ball in the hoop")
    parser.add_argument(
        "--response-file",
        type=Path,
        help="Parse this saved response instead of contacting the NVILA server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: 3dway_policy/outputs).",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Open the saved trajectory over scene.ply after inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    example_dir = Path(__file__).resolve().parents[1] / "example_data"
    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "outputs"
    cameras = load_camera_rig(example_dir / "camera_parameters.json")
    policy = NVILAPolicy(
        model_name=args.model or "offline-response",
        base_url=args.base_url,
    )

    images = []
    for image_name in ("view1.png", "view2.png"):
        with Image.open(example_dir / image_name) as image:
            images.append(image.convert("RGB"))

    if args.response_file is not None:
        prediction = policy.prediction_from_response(
            args.response_file.read_text(encoding="utf-8"), cameras.values()
        )
    else:
        if not args.model:
            raise SystemExit("--model is required unless --response-file is provided.")
        prediction = policy.predict(images, cameras.values(), args.instruction)

    artifacts = save_prediction_outputs(
        prediction,
        images,
        output_dir,
        metadata={
            "instruction": args.instruction,
            "model": args.model,
            "view_mapping": {name: camera.source_camera for name, camera in cameras.items()},
        },
    )
    print(
        json.dumps(
            {
                "prediction": str(artifacts.result_json),
                "view_images": [str(path) for path in artifacts.view_images],
            },
            indent=2,
        )
    )

    if args.visualize_3d:
        visualize_prediction_3d(example_dir / "scene.ply", artifacts.result_json)


if __name__ == "__main__":
    main()
