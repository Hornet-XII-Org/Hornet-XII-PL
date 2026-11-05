# Depth estimation node 

**ROS package:** `hornet_depth` · **executable:** `depth_estimation_node` · **owner:** to assign for integration, uncertainty tuning and pool validation.

`hornet_depth/geometry.py` implements sparse stereo triangulation, pixel-noise covariance, rigid transforms and source-selection policy without importing ROS. `node.py` adapts these to ROS topics and timestamped TF. `config/depth.yaml` contains initial gates/switch thresholds. `launch/depth.launch.py` accepts real left/right calibration paths.

The saved ROS 2 Jazzy tutorial pair is included in `front_camera/config/measured/`. See its [geometry, provenance and launch commands](../front_camera/config/measured/README.md). Select the same pair for rectification, detection metadata and depth estimation; the synthetic examples remain the launch defaults and are disabled by default.

## Contract

Inputs per class: `/front/detections/<type>` and `/front/color/<type>`, both `hornet_interfaces/msg/StereoObservation`. Outputs: `/objects/<type>/position` (`ObjectPosition`) and `/perception/selection/<type>` (`SourceSelection`). Reliable, volatile, KEEP_LAST depth 10. See [the full contract](../docs/interfaces.md).

The node loads YAML once, verifies a horizontal jointly rectified pair, and checks every observation's calibration ID and left rectified frame. It expects already matched left/right centers for the same physical object. It does not run ML, segmentation, stereo association, dense disparity, or a point cloud.

## Processing

1. Validate class/source/instance, confidence, finite pixels, bounds, timestamp and calibration identity.
2. Apply corrected-disparity, positive-depth, range and reprojection consistency gates; triangulate with both saved P matrices.
3. Propagate independent per-coordinate pixel noise through triangulation to a 3×3 covariance. Its largest principal-axis standard deviation is used for selection quality.
4. Maintain candidates per `(class, instance_id, method)`, reject out-of-order observations, expire stale candidates, and avoid comparing frames separated by more than the source-skew window.
5. Prefer ML far away, color at/below the enter threshold, and keep color until the exit threshold is reached. Use a valid alternative when the preferred method is absent or too uncertain; withhold positions if neither is usable.
6. Request TF from the left rectified optical frame to `map` at the image timestamp. Rotate/translate XYZ and rotate covariance. If TF is missing, report feedback and retry while the observation is fresh. Never publish camera XYZ with a `map` label.
7. Publish at most one map position per increasing image timestamp per instance. Expire tracks and cap memory. Feedback is emitted on new measurements/state changes, including loss of valid observations.

Both producer methods need stable shared instance IDs. Their centers must represent corresponding physical points; unrelated bounding-box centers can produce biased depth even with low reprojection residual. Neither confidence nor reprojection consistency proves association is correct.

## What uncertainty means

`position_stddev_m` is a stereo estimate based on `pixel_stddev_px`. The covariance does not include calibration, TF/pose/extrinsic uncertainty, center-definition bias, or incorrect correspondence. `includes_localization_uncertainty=false` makes this explicit. This is a useful selection signal, not actual ground-truth position error. Validate the supplied pixel-noise model against labelled/pool measurements.

Map transformation needs localization and extrinsics. Until those placeholder roles are implemented, no trustworthy map position can be produced. IMU+depth alone does not provide absolute X/Y, and a fixed frame name does not remove drift. See [frame conventions](../docs/frames.md).

## Configuration

All values are read at startup; restart after changing them. Main defaults are enter/exit color at 2.0/2.5 m, max stereo standard deviation 0.5 m, observation expiry 0.5 s, source time-skew 0.15 s, and 0.2–15 m valid range. They are tunable starting values, not validated camera capabilities. Example calibrations require an explicit `allow_example_calibration=true` for tests and remain unsuitable for deployment.

## Acceptance criteria

- Producer contracts validated in Jazzy, including multiple same-class instances.
- Known 3D points and independent measured ranges recovered within the agreed test tolerance.
- Hysteresis, fallback, no-match, stale data, duplicate timestamps, missing TF and wrong calibration tested.
- Stationary target stays fixed in `map` while the robot moves, within measured localization/calibration limits.
- Full robot pose uncertainty is accounted for downstream before fusing map object observations.
