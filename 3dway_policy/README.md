# 3DWay Policy

`3dway-policy` is the lightweight inference and geometry package for
**3DWay: Generalizing Robot Manipulation via 3D Consistent Waypoints**. It
connects to the repository's OpenAI-compatible NVILA server, parses
multi-view-consistent 2D trajectories, triangulates them in the world frame,
and exports visualization artifacts and pose-plus-gripper actions.

For the method, training recipe, and paper results, see the
[repository README](../README.md).

## Data flow

```text
two RGB views + language instruction + camera calibration
                            │
                            ▼
              NVILA paired 2D trajectories
                            │
                            ▼
                  geometric triangulation
                            │
                            ▼
       3D waypoints + ray residuals + Nx8 robot actions
                            │
                            ▼
             JSON + two 2D overlays + Open3D view
```

## Installation

From the repository root:

```bash
python -m pip install -e ./3dway_policy
```

Python 3.10 or newer is required. Install the optional Open3D dependency when
3D visualization is needed:

```bash
python -m pip install -e './3dway_policy[visualization]'
```

## Quick start

### 1. Start the NVILA server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_server_nvila.sh /path/to/3DWay-15B
```

The server listens on `http://0.0.0.0:8888` by default. Configure `HOST`,
`PORT`, or `CONV_MODE` through environment variables if needed.

### 2. Run the paired-view example

In another shell:

```bash
python 3dway_policy/examples/run_policy.py \
  --model 3DWay-15B \
  --instruction "put the ball in the hoop"
```

`--model` must match the checkpoint directory name printed by the server. Use
`--base-url` when serving from another machine:

```bash
python 3dway_policy/examples/run_policy.py \
  --model 3DWay-15B \
  --base-url http://<server-ip>:8888 \
  --instruction "put the ball in the hoop"
```

The example uses:

```text
example_data/
├── camera_parameters.json  # Calibrated view1/view2 camera rig
├── scene.ply               # Point cloud for optional 3D visualization
├── view1.png               # Left-shoulder camera
└── view2.png               # Right-shoulder camera
```

## Outputs

Each online or offline run writes:

```text
outputs/
├── prediction.json
├── view1_trajectory.png
└── view2_trajectory.png
```

The output directory is overwritten on subsequent runs. Use
`--output-dir <directory>` to retain multiple predictions.

`prediction.json` contains:

| Field | Shape/type | Description |
| --- | --- | --- |
| `response_text` | string | Raw tagged response returned by NVILA |
| `trajectories_2d` | two `N × 3` arrays | Normalized `(x, y, gripper)` trajectories |
| `trajectory_3d` | `N × 3` | Triangulated XYZ waypoints in the world frame |
| `ray_residuals` | `N` | Distance between the two closest camera-ray points |
| `actions` | `N × 8` | `[x, y, z, qx, qy, qz, qw, gripper]` |
| `inference_seconds` | float or null | Client-observed online inference latency |
| `metadata` | object | Instruction, model name, and view mapping |
| `artifacts` | object | Relative paths to the two rendered overlays |

The 2D overlays use a blue-to-red trajectory to indicate temporal direction.
Blue markers denote opening the gripper and red markers denote closing it.

## Input contract

### Model response

The parser requires a tagged trajectory for both views:

```text
<ans_view1>[(0.57, 0.80), <action>Close Gripper</action>, ...]</ans_view1>
<ans_view2>[(0.21, 0.76), <action>Close Gripper</action>, ...]</ans_view2>
```

Both views must contain the same number of points and matching gripper states.
Coordinates are normalized by image width and height and are expected to lie in
the image domain.

### Camera calibration

Calibration uses schema version 1. See the complete
[`camera_parameters.json`](example_data/camera_parameters.json) example.

| Key | Meaning |
| --- | --- |
| `schema_version` | Must be `1` |
| `extrinsics_convention` | Must be `camera_to_world` |
| `views.<name>.width` / `height` | Image dimensions in pixels |
| `views.<name>.intrinsics` | 3×3 pinhole-camera intrinsic matrix |
| `views.<name>.camera_to_world` | 4×4 homogeneous camera-to-world transform |
| `views.<name>.source_camera` | Optional source-camera label |

The implementation accepts negative focal lengths when they are part of the
source camera convention, as in the included RLBench example.

The order of supplied images must match the order of views in the calibration
file. The checked-in example maps `view1` to `left_shoulder` and `view2` to
`right_shoulder`.

## Python API

```python
from pathlib import Path

from PIL import Image
from three_d_way_policy import NVILAPolicy, load_camera_rig

data_root = Path("3dway_policy/example_data")
cameras = load_camera_rig(data_root / "camera_parameters.json")
images = [
    Image.open(data_root / "view1.png").convert("RGB"),
    Image.open(data_root / "view2.png").convert("RGB"),
]

policy = NVILAPolicy(
    model_name="3DWay-15B",
    base_url="http://127.0.0.1:8888",
)
prediction = policy.predict(
    images,
    cameras.values(),
    "put the ball in the hoop",
)

print(prediction.trajectory_3d)
print(prediction.actions)
```

The default action conversion assigns the fixed XYZW quaternion
`(0, 1, 0, 0)` to every waypoint. Gripper state is encoded as `0` for closed
and `1` for open.

## Offline replay

To test parsing, triangulation, and visualization without a running model
server, save a raw model response to a text file:

```bash
python 3dway_policy/examples/run_policy.py \
  --response-file /path/to/response.txt \
  --output-dir 3dway_policy/outputs/offline-example
```

## 3D visualization

Overlay a saved prediction on the included point cloud:

```bash
python 3dway_policy/examples/visualize_3d.py \
  --result 3dway_policy/outputs/prediction.json \
  --scene 3dway_policy/example_data/scene.ply
```

Or display the Open3D window immediately after inference:

```bash
python 3dway_policy/examples/run_policy.py \
  --model 3DWay-15B \
  --visualize-3d
```

The Open3D view renders the source point cloud, the triangulated 3D polyline,
and markers at gripper-state transitions.


## Safety and scope

This package produces geometric waypoints, not hardware-safe motor commands.
Before real-robot execution, add robot-specific inverse kinematics, workspace
limits, collision checking, velocity and acceleration limits, and an emergency
stop. The paper's direct-execution controller uses task-specific assumptions
such as a fixed top-down orientation and is intended for suitable simple tasks.
