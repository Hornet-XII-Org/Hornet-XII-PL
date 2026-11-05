# Hornet XII PL

**Perception and Localization · ROS 2 Jazzy · subteam development repository**

This repository organizes the Hornet XII P&L framework into component areas. Each top-level component folder can contain several nodes, tools and configuration files, with its interfaces and responsibilities described in a README. `hornet_interfaces/` is a shared message package, not another node.

Inspired by the modular camera architecture in [Hornet 9](https://github.com/hornetxauv/h9_camera_ws/tree/b2f5b65507e74cbada62e6a44cd9a10d4445c809). This code is newly written for Jazzy and the framework below. It does not copy Hornet 9's machine-specific paths, older launch wiring, model weights, or calibration measurements.

## Overview

![Hornet XII PL system overview](docs/images/system-overview.png)

## Framework

![Hornet XII perception and localization architecture](docs/images/framework.png)

[Scalable SVG](docs/images/framework.svg) · [System overview](docs/images/system-overview.svg) · [Editable diagram source](docs/architecture.mmd)

1. The front camera driver publishes one side-by-side stereo image. The front calibration node splits and rectifies it with previously measured left/right YAML files.
2. Object detection and front color thresholding each find **associated centers in both rectified images**, then publish one topic per object class.
3. Depth estimation triangulates the centers, estimates stereo uncertainty, selects the preferred method, and transforms the result into `map` at the image timestamp. It publishes one position topic and one selection-feedback topic per class.
4. The bottom color camera has its own calibration node and saved YAML in `bottom_camera/`. Its rectified BGR8 image feeds optical flow and bottom color thresholding.
5. The localization owner fuses IMU/depth and metric optical-flow information, maintains the pose/TF chain, and defines the map origin. Bottom object positions additionally require a known plane or altitude and correct camera geometry.

Dashed arrows represent configuration reads or feedback. Focal length and stereo baseline are saved in YAML and read once by consumers; there are **no continuously published focal-length/baseline topics**. “Calibration” in the runtime graph means applying saved calibration. A separate offline tool estimates it before deployment.

**Map coordinates are metric coordinates in a fixed frame, not camera-relative XYZ and not values scaled to 0–1.** Expressing a point in `map` does not remove localization drift or calibration error. The current covariance describes pixel-induced stereo uncertainty conditional on exact robot pose/calibration; it is not a ground-truth error bound.

## Component responsibilities

| Folder | Responsibility |
|---|---|
| [front_camera/](front_camera/README.md) | Front camera acquisition, stereo calibration and rectification |
| [bottom_camera/](bottom_camera/README.md) | Bottom color camera acquisition, calibration and rectification |
| [object_detection/](object_detection/README.md) | ML detection, stereo matching, stable instance identity |
| [front_color_thresholding/](front_color_thresholding/README.md) | Per-object HSV segmentation and stereo centers |
| [depth_estimation/](depth_estimation/README.md) | Sparse stereo geometry, uncertainty, source selection, map transform |
| [optical_flow/](optical_flow/README.md) | Metric bottom-camera motion with scale and covariance |
| [sensor_bridge/](sensor_bridge/README.md) | IMU, magnetic field, pressure/depth frames, units and timestamps |
| [localization/](localization/README.md) | EKF, map management, sensor fusion, TF and localization uncertainty |
| [bottom_color_thresholding/](bottom_color_thresholding/README.md) | Bottom object segmentation and map-plane localization |
| [hornet_interfaces/](hornet_interfaces/README.md) | Shared message contracts; changes reviewed by producer and consumer owners |

Members should begin with [CONTRIBUTING.md](CONTRIBUTING.md), [topic contracts](docs/interfaces.md), and their folder README. Use the [node work issue template](.github/ISSUE_TEMPLATE/node-work.md) to assign work. Integrate node packages into your team's robot workspace.

The [measured front stereo calibration](front_camera/config/measured/README.md) from the ROS 2 Jazzy tutorial is included with commands for rectification and depth estimation. It describes 320×240 per lens (640×240 side-by-side capture); use it only with the corresponding camera rig and capture settings.

## Topic contract at a glance

`<type>` is one of `gate`, `yellow_flare`, `orange_flare`, `blue_flare`, `red_flare`, `blue_drum`, `red_drum`. Edit the depth YAML list and producer implementations together when the competition classes are finalized.

| Topic | Message | Producer → consumer |
|---|---|---|
| `/front/image_raw` | `sensor_msgs/msg/Image` | Front driver → front calibration instance |
| `/front/left/image_rect`, `/front/right/image_rect` | `sensor_msgs/msg/Image` | Front calibration → both detection methods |
| `/front/detections/<type>` | `hornet_interfaces/msg/StereoObservation` | Object detection → depth estimation |
| `/front/color/<type>` | `hornet_interfaces/msg/StereoObservation` | Front color thresholding → depth estimation |
| `/objects/<type>/position` | `hornet_interfaces/msg/ObjectPosition` | Depth estimation → mission/mapping consumers |
| `/perception/selection/<type>` | `hornet_interfaces/msg/SourceSelection` | Depth estimation → both detection methods |
| `/bottom/image_raw` → `/bottom/image_rect` | `sensor_msgs/msg/Image` | Bottom driver → calibration → flow / bottom color |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | Localization → consumers |
| `/bottom/objects/<type>/position` | `hornet_interfaces/msg/ObjectPosition` | Bottom color thresholding → mission/mapping consumers |

Each stereo observation contains the class, source, calibration ID, physical-instance ID, four pixel coordinates, acquisition timestamp, confidence, and pixel uncertainty. A class can have several instances; each is a separate message. Both methods must share instance identities before the depth node can compare them. The depth node does not infer correspondence between arbitrary left/right detections.

## Integration and simulation

Coordinate component interfaces using [topic contracts](docs/interfaces.md) and [frame conventions](docs/frames.md). Simulation scenarios can validate image timing, camera geometry, source handover, robot localization and map-coordinate object positions. Simulation tooling belongs to the team's chosen simulator integration.

## Storage and contribution rules

- Version node source, schemas, small configuration, documentation, and measured YAMLs with provenance.
- Keep bags, calibration captures, datasets, model weights, and generated build/install/log outputs out of Git. Document their storage location and version instead.
- Preserve image timestamps and use the shared frame/topic contracts. Changes need review from both sides of an interface.
- Do not commit access tokens. Manage repository access and member invitations through the subteam lead.
- No open-source license has been chosen for this team project. Package metadata uses `LicenseRef-Proprietary`; replace the maintainer contact placeholder before releasing packages.

See [architecture and frame conventions](docs/frames.md), [implementation notes](docs/design-decisions.md), and [primary references](docs/references.md).
