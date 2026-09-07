#!/usr/bin/env python3
"""Visualize a saved 3DWay prediction on the example point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

from three_d_way_policy import visualize_prediction_3d


def main() -> None:
    policy_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=policy_root / "outputs" / "prediction.json",
        help="Prediction JSON produced by run_policy.py.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=policy_root / "example_data" / "scene.ply",
    )
    parser.add_argument("--marker-radius", type=float, default=0.015)
    args = parser.parse_args()

    visualize_prediction_3d(
        args.scene,
        args.result,
        marker_radius=args.marker_radius,
    )


if __name__ == "__main__":
    main()

