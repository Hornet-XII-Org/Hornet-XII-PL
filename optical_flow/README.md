# Bottom-camera metric optical flow

[optical_flow_node.cpp](src/optical_flow_node.cpp) tracks Shi–Tomasi corners with pyramidal Lucas–Kanade optical flow and publishes **robot-frame velocity in metres per second** on `/bottom/flow/odometry`. The ROS package is `optical_flow`; its executables are `optical_flow_node` and `of_video_player`.

Normal operation uses rectified images, saved bottom-camera calibration, known floor distance with uncertainty, and timestamped robot orientation. It **withholds odometry** when required inputs are missing or unusable. The current configuration contains the explicitly assumed video-test model described below, not measured camera calibration. The opt-in fixed-orientation mode replaces the external attitude input only for the level, nonrotating test.

The source files are [optical_flow_node.cpp](src/optical_flow_node.cpp) and [of_video_player.cpp](src/of_video_player.cpp). There are no pixel-per-second velocity outputs in the interface.

## Run the current video test

Stop older player/detector instances first. In a terminal with your ROS 2 distribution sourced:

```bash
cd ~/Hornet_OF
colcon build --packages-select optical_flow --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo &&
source install/setup.bash &&
ros2 launch optical_flow optical_flow.launch.py \
  play_video:=true loop:=true input_reliable:=true \
  assume_fixed_orientation:=true
```

In a second sourced terminal, open the annotated video:

```bash
cd ~/Hornet_OF
source install/setup.bash
ros2 run optical_flow view_video.py
```

In a third sourced terminal, inspect body-frame velocity in **m/s**:

```bash
cd ~/Hornet_OF
source install/setup.bash
ros2 topic echo /bottom/flow/odometry --field twist.twist.linear
```

The overlay should say **`TEST fixed attitude`** when a usable velocity estimate exists. The test assumes a level, nonrotating body, a camera pointing straight down with image-up = robot-forward, no distortion, 1 m height with 0.1 m standard deviation, coincident camera/body origins, and generated `fx = fy = 320 px` at 640×360. Body X is forward, Y is left, and Z is up. These speeds are model-based estimates; the generated focal length does not establish the actual video's metric accuracy.

The current [bottom_camera.yaml](config/bottom_camera.yaml) has both confirmation flags set to `1`, allowing the assumed model to be evaluated. These flags do not prove that the calibration is measured or correct. The fixed-attitude switch does not enable missing calibration or bypass other validity checks. Stop and rebuild/re-source after code changes; restart after editing configuration. For real camera/IMU operation, use `assume_fixed_orientation:=false` and supply measured calibration, mounting and timestamped orientation.

## Build and launch

In a sourced ROS 2 environment, run from the workspace root. The package folder is `~/Hornet_OF/optical_flow`, and its ROS package name is `optical_flow`:

```bash
cd ~/Hornet_OF
rosdep install --from-paths optical_flow --ignore-src -r -y &&
colcon build --packages-select optical_flow --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo &&
source install/setup.bash &&
ros2 launch optical_flow optical_flow.launch.py
```

After renaming the package folder, open the new folder in your editor and rebuild from a fresh terminal with your ROS 2 distribution sourced. If an earlier build cached the old source path, rebuild with `colcon build --packages-select optical_flow --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo`. Old build/install artifacts are not moved or deleted by the source-folder rename.

This starts the estimator and waits for your camera and orientation sources. It does not automatically play a video or manufacture orientation measurements.

Edit [config/optical_flow.yaml](config/optical_flow.yaml) and supply a measured calibration file:

```bash
ros2 launch optical_flow optical_flow.launch.py \
  params_file:=/absolute/path/to/optical_flow.yaml \
  calibration_file:=/absolute/path/to/bottom_camera.yaml
```

Or run just the executable:

```bash
ros2 run optical_flow optical_flow_node --ros-args \
  --params-file /absolute/path/to/optical_flow.yaml \
  -p calibration_file:=/absolute/path/to/bottom_camera.yaml
```

Restart the node after changing calibration or configuration. Launch arguments `input_reliable`, `assume_fixed_orientation`, and `opencv_threads` override the corresponding YAML settings; their defaults are `false`, `false`, and `0`.

## Inputs

| Topic | Type | Contract |
| --- | --- | --- |
| `/bottom/image_rect` | `sensor_msgs/msg/Image` | Already rectified, `mono8`, `bgr8`, or `rgb8`; acquisition timestamp; calibrated optical frame and exact calibrated dimensions |
| `/bottom/orientation` | `geometry_msgs/msg/PoseWithCovarianceStamped` | Required in normal mode: body orientation in the fixed reference frame at acquisition time; unit quaternion and known positive orientation covariance. Ignored in fixed-orientation test mode. |

The image subscription defaults to **ROS sensor-data QoS**, including best-effort reliability and depth 5. The orientation subscription also uses sensor-data QoS. Topic names can be changed with normal ROS remapping.

For the orientation message:

- `header.frame_id` must equal `reference_frame` (default `odom`). This frame must have **+Z up**, with the static horizontal floor parallel to its XY plane.
- `pose.pose.orientation` rotates coordinates from `child_frame` (default `base_link`) into `reference_frame`. `PoseWithCovarianceStamped` has no child-frame field: the source must explicitly follow this configured body-frame contract. Convert an IMU's mounting, world convention, and reference frame upstream as necessary.
- The bottom-right 3×3 block of `pose.covariance` (indices 21–23, 27–29, 33–35) contains orientation covariance in radians². It must be finite, symmetric and positive definite. Zero/unknown covariance and invalid quaternions are rejected. Position and its covariance are not used.
- Orientation is sampled at **both image timestamps** using quaternion SLERP between bracketing samples; it is never extrapolated or replaced with the latest available attitude. The maximum bracket span is `orientation_max_gap_s` (default 0.1 s).

In normal orientation mode, images wait up to `orientation_wait_s` (default 0.15 s) for matching orientation. The pending queue is capped at `max_pending_images` (default 5). If orientation never arrives, the image still appears on `/bottom/flow/image` with corner tracking and a **preview-only** message; metric odometry is withheld and validity is false. Missing orientation breaks the metric pair, so motion is only published again after two consecutive usable images have matching attitude. Overflow also resets tracking. All streams must use the same acquisition clock, including during bag/video replay.

## Saved bottom calibration and metric scale

Fill [config/bottom_camera.yaml](config/bottom_camera.yaml), retaining its OpenCV YAML matrix format:

| Field | Meaning |
| --- | --- |
| `calibrated`, `extrinsics_calibrated` | Set each to `1` only after replacing its placeholder values with measured calibration |
| `image_width`, `image_height` | Exact dimensions of `/bottom/image_rect` |
| `rectified_camera_matrix` | 3×3 rectified pinhole matrix: left 3×3 of CameraInfo **P**, including focal lengths and principal point |
| `camera_frame` | Image optical frame: X right, Y down, Z into scene |
| `body_frame` | Must equal output `child_frame` |
| `camera_to_body_quaternion_xyzw` | Measured rotation from rectified optical coordinates into body coordinates |
| `camera_position_body_m` | Camera optical centre relative to the body origin, expressed in body coordinates, in metres |

Use underwater/housing calibration for the real AUV or the simulator's actual camera model. Raw distorted CameraInfo K and rectified P can differ. The node does not undistort or rectify images itself. When rectification introduces an additional camera-frame rotation, include it in the saved optical-to-body transform.

In [config/optical_flow.yaml](config/optical_flow.yaml), set:

- `body_height_m`: **body origin's constant vertical distance above the floor**, in metres. This is not depth below the surface and is not necessarily camera height. The camera's rotated mounting offset is added at each exposure.
- `height_std_m`: positive standard deviation of that height, in metres. Zero means unknown uncertainty and disables metric output.
- Measured/defensible uncertainty values for `focal_relative_std`, `principal_point_std_px`, `extrinsic_angle_std_rad`, and `lever_arm_std_m`. Their defaults are tuning assumptions, not calibration results.

The current test configuration uses the requested 1.0 m camera-to-floor height and 0.1 m standard deviation. `body_height_m: 1.0` is provisional: it assumes the camera and body origins coincide. Confirm the camera offset before using this as body-origin height. For a level body, `body_height = camera_height - camera_position_body_m[2]`.

The saved test model assumes no distortion, a centred principal point, `fx = fy = 320 px` at 640×360 (approximately 90° horizontal FOV), and a downward camera with image-up pointing robot-forward. Its mounting rotation maps image-right to body-right, image-up to body-forward, and the optical axis to body-down. The focal length was generated for a hypothetical camera; it was not recovered from the video. Calibration confirmation flags must only be enabled when actual scale and the camera-to-body translation have been verified; flags alone do not turn the generated focal length into a measured calibration. Assuming no distortion alone does not determine focal length or metres per pixel. Acquisition-time robot orientation messages are required in normal mode; the fixed camera mounting rotation does not replace them. The explicit fixed-orientation test mode supplies assumed attitude instead.

At 640×360, use calibration for 640×360. If scaling a 1920×1080 rectified image by one third, scale both focal lengths and both principal-point coordinates consistently with the image resampling convention. The node rejects size mismatches instead of silently rescaling calibration. Tracking may internally downsample larger images to 640 pixels wide; its tracks are mapped back to input-image coordinates before using the saved calibration.

## How the velocity becomes m/s

The ROS-independent helpers are:

- [flow_tracker.hpp](include/optical_flow/flow_tracker.hpp): persistent Shi–Tomasi corners, pyramidal Lucas–Kanade tracking, forward/backward rejection, and feature replenishment. Shi–Tomasi detects general intensity corners, not exclusively pool-line intersections.
- [metric_geometry.hpp](include/optical_flow/metric_geometry.hpp): validated quaternions, timestamp interpolation, and rotation operations.
- [bottom_calibration.hpp](include/optical_flow/bottom_calibration.hpp): saved calibration loading and validation.
- [metric_estimator.hpp](include/optical_flow/metric_estimator.hpp): floor projection, robust translation consensus, camera mounting-offset compensation, metric conversion, and covariance.

For each feature and exposure, the estimator forms an optical ray `K_rect^-1 [u,v,1]`, rotates it into the fixed reference frame using timestamped body orientation and the saved camera mounting rotation, then intersects it with the horizontal floor. These intersections are **in metres**, because the ray is scaled by known camera height.

For a stationary floor point:

```text
camera displacement = previous floor offset - current floor offset
body displacement   = camera displacement - rotation-induced camera mounting-offset motion
body velocity       = R_reference_body(current)^T * [dx/dt, dy/dt, 0]
```

`dt` comes from the two image acquisition timestamps. Robust consensus is fitted **after rotation compensation in metric space**, requiring sufficient inlier count, a majority inlier ratio and two-dimensional spatial support. Incoherent flow, rays near the horizon, and speeds above `max_speed_mps` are rejected.

For a level camera with negligible mounting offset, the familiar camera-frame special case is:

```text
vx_camera = -du * camera_height_m / (fx_px * dt_s)
vy_camera = -dv * camera_height_m / (fy_px * dt_s)
```

The negative sign accounts for the floor moving opposite the camera. The production estimator also accounts for camera roll/pitch/yaw and transforms the result into the declared body frame. No pixel-per-second velocity topic is published.

## Output and covariance contract

| Topic | Type | Meaning |
| --- | --- | --- |
| `/bottom/flow/odometry` | `nav_msgs/msg/Odometry` | Valid metric velocity, with acquisition timestamp and covariance |
| `/bottom/flow/valid` | `std_msgs/msg/Bool` | Whether the latest processed frame produced usable metric odometry |
| `/bottom/flow/image` | `sensor_msgs/msg/Image` | Viewing stream: corners, accepted metric inlier tracks, and body X/Y speed in m/s, or a preview-only status |
| `/bottom/flow/assumed_orientation` | `std_msgs/msg/Bool` | Whether fixed test attitude is enabled; reliable, transient-local status |

Odometry and validity use reliable QoS. The annotated image publisher uses reliable QoS with depth 5; the viewer requests reliable delivery with depth 2. The input remains sensor-data QoS by default, as required by the camera interface.

For each published odometry message:

- `header.stamp` is the **current image acquisition stamp**, preserved exactly. Velocity is the average over the preceding accepted image interval, expressed in the current body orientation.
- `header.frame_id = reference_frame`; `child_frame_id = child_frame`. Linear twist and its covariance are expressed in `child_frame`, following [nav_msgs/Odometry semantics](https://github.com/ros2/common_interfaces/blob/rolling/nav_msgs/msg/Odometry.msg).
- `twist.twist.linear` contains m/s. The observed components are the reference-frame horizontal velocity projected into the body frame. Full 3D translation is not observed.
- The full 3×3 linear covariance is rotated into the body frame, including off-diagonal terms. It incorporates residual dispersion, pixel uncertainty, height uncertainty, orientation covariance, focal/principal-point uncertainty, mounting-angle uncertainty, mounting-position uncertainty, and a nonzero velocity noise floor. Systematic uncertainties do not vanish with more corners. Units are `(m/s)^2`.
- The unobserved reference vertical direction retains `unobserved_variance` (default `1e6`) after rotation into the body frame. Thus a displayed/serialized zero along an unobserved direction is **not** a confident zero measurement. If tilted, the large uncertainty is spread across correlated body axes.
- Angular velocity is not measured: angular twist is a placeholder with large covariance on all three axes. Position is not integrated: pose position is a placeholder, orientation is copied from the synchronized input (or the explicitly assumed identity attitude in test mode), and all pose axes have large covariance. **Fuse only the supported linear-velocity subspace, not pose or angular twist.** No TF is broadcast.

There is no odometry on initialization, unknown scale/calibration, missing/stale orientation in normal mode, feature loss, frame mismatch, invalid timestamps, excessive image gaps, invalid geometry or failed robust fitting. `/bottom/flow/valid` is false for rejected frames and becomes false after 0.5 seconds without input. In test mode, a true validity flag means the configured assumed model passed the estimator checks; it does not certify physical metric accuracy. Consumers must also enforce their own acquisition-timestamp timeout; stopped nodes cannot keep publishing validity. Small zero velocity with positive covariance is allowed only when actual image tracks support stationarity.

Covariance is a conservative first-order model, not a guarantee against wrong associations, incorrect calibration, a sloped/nonplanar floor, moving shadows/objects, refraction errors or time synchronization errors. Validate and tune it using ground truth before navigation use. Changes in body height violate the current constant-height model; a timestamped altitude/range source is a future extension.

## Video playback and viewing

The global `VIDEO_PATH` near the top of [of_video_player.cpp](src/of_video_player.cpp) selects the default video. Change it and rebuild, or supply `video_path:=/absolute/path/to/video.MOV` without rebuilding. Optional playback is enabled explicitly:

```bash
ros2 launch optical_flow optical_flow.launch.py play_video:=true loop:=true input_reliable:=true
```

**Only use this as a rectified input if the selected video has already been rectified using matching calibration.** The player resizes to 640×360 for the supplied 16:9 videos before publication. It publishes `video_input`, remapped by the launch file to `/bottom/image_rect`. It does not provide calibration, actual body orientation or fabricated odometry. A plain video without matching acquisition-time orientation and metric scale cannot produce valid odometry through this interface. If calibration is disabled, corner-tracking preview remains available while metric output is withheld.

For the explicitly assumed **level, nonrotating video test**, use:

```bash
ros2 launch optical_flow optical_flow.launch.py \
  play_video:=true loop:=true input_reliable:=true assume_fixed_orientation:=true
```

`assume_fixed_orientation` is **false by default**. Enabling it supplies an identity body-to-reference quaternion at each image's acquisition timestamp inside the estimator, without requiring or publishing `/bottom/orientation` messages. The body must remain level with fixed zero yaw relative to the configured reference. Incoming orientation messages are ignored in this explicit mode; there is no automatic fallback to assumed attitude in normal operation. `fixed_orientation_std_rad` defaults to one degree as a test uncertainty assumption and must remain finite and positive. A startup warning and `TEST fixed attitude` overlay identify these results; the reliable transient-local `/bottom/flow/assumed_orientation` Bool topic reports whether this assumption is enabled.

This mode bypasses only the external-orientation requirement. Calibration confirmation flags, metric scale/height and uncertainty, matching image dimensions/frames, valid acquisition timing, track quality, robust motion consensus and covariance checks still apply. Unknown scale and featureless images still produce no odometry. It does not fabricate zero translation. With the generated focal length, displayed m/s values are **approximate model-based test results**, not validated AUV measurements. Return to `assume_fixed_orientation:=false` when using measured attitude.

A bag containing rectified images and matching orientation is preferable for quantitative replay. Video timestamps are generated from the source frame index and FPS, anchored to the node's clock at startup. They are not an original recorded orientation timeline. Looping inserts a gap to force a tracker reset, so external orientation must follow the replay timeline too.

Use separate sourced terminals for the viewer and topic inspection, as shown in the video-test instructions above.

To inspect only metric velocity, run `ros2 topic echo /bottom/flow/odometry --field twist.twist.linear`. Values are in **m/s in `child_frame_id`**, not pixel/s. The frame convention follows the [ROS Odometry message definition](https://github.com/ros2/common_interfaces/blob/rolling/nav_msgs/msg/Odometry.msg). If no odometry messages appear, check the node warning: calibration/scale settings, tracking, image frame/dimensions and timestamps must be valid, and normal mode additionally requires synchronized orientation. Fixed test mode bypasses only that external-orientation requirement.

In normal mode, if a warning says `No bracketing orientation at image acquisition time`, the node has no valid exact/bracketing attitude samples on `/bottom/orientation` for that image. With the preview fix, this warning withholds odometry but does not prevent corner-tracking video from appearing after the short orientation wait. Video playback does not generate orientation. For the agreed nonrotating video assumption, launch with `assume_fixed_orientation:=true`; this must be passed explicitly even if the YAML has been edited, because the launch argument overrides it. As a diagnostic, view the input directly with `ros2 run optical_flow view_video.py --ros-args -p topic:=/bottom/image_rect`. A blank input view indicates a separate video/publication problem.

The viewer requires a graphical desktop; press `q` to close it. Its default topic is now `/bottom/flow/image`. Standard ROS image viewers can also subscribe to that topic.

The player retains reliable publication. The optional `input_reliable:=true` launch argument also makes the detector request reliable input delivery. This is useful for the delivery-loss issue observed earlier in this session: playback reached 30 publish calls/s, while detection received about 3.5 frames/s despite 9 ms callbacks, and reliable delivery resolved the observed flicker. Both producer and consumer must support the requested reliability; a reliable subscription does not match a best-effort camera publisher. Keep the default sensor-data subscription for best-effort cameras. Reliable retransmission can add latency and cannot fix insufficient bandwidth.

Other launch options: `image_width` (640 by default, 0 preserves the source), `opencv_threads` (0 retains OpenCV's default), `params_file`, and `calibration_file`. The player-only `fps` ROS parameter changes the source timeline if overridden. The source clip `straight_line.MOV` is 30 FPS; processing faster does not create additional captured frames.

The node logs processed-image rate, valid metric odometry rate, and processing time every two seconds. A high image rate with zero odometry can be intentional withholding; read the warning reason. At 30 FPS each image has about 33 ms of processing budget. Raw BGR 640×360 at 30 FPS carries about 21 MB/s per image topic before middleware overhead. Do not infer processing FPS solely from `ros2 topic hz`, which measures reception at that subscriber.

## Quantitative tests

```bash
cd ~/Hornet_OF
colcon test --packages-select optical_flow
colcon test-result --verbose
```

The test suite includes:

- [test_metric_estimator.cpp](test/test_metric_estimator.cpp): known metric translation/sign/scale; timestamp-to-m/s conversion; combined tilt/yaw/translation; pure rotation with a nonzero camera mounting offset; body-frame covariance including the unobserved direction; covariance sensitivity to height/focal uncertainty; noise plus 25% outliers over 30 trials; and explicit rejection of unknown calibration/scale/uncertainty, bad orientation, nonfinite tracks, incoherent motion, degenerate features and upward rays. It also checks opt-in fixed-attitude timestamps/uncertainty and rejects invalid test assumptions; image tracking through metric conversion, exact/interpolated orientation timing, quaternion normalization, and rejects extrapolation/stale brackets.
- [test_odometry_contract.py](test/test_odometry_contract.py): launches the **actual C++ node** in an isolated ROS domain, drives stamped synthetic images and orientation, and checks metric values, output timestamp, frame IDs, positive covariance and unobserved-axis covariance. It also verifies timestamp-preserving, nonblank preview output without startup orientation and no metric estimate across a missing-attitude image. It also tests fixed-attitude output without orientation messages while retaining unknown-height and blank-image rejection. It checks that unknown scale, stale orientation in normal mode, blank images, calibration-size mismatch, duplicate/backward timestamps and excessive image gaps produce no odometry, followed by feature recovery.
- [test_flow_tracker.cpp](test/test_flow_tracker.cpp): synthetic image translations, stationary tracking, internal-resize coordinate restoration, resets, feature loss/recovery, size changes and supplied-video track availability.
- [test_image_message.cpp](test/test_image_message.cpp): image/timestamp round trips, non-contiguous images, row padding, mono/RGB conversion and malformed payload rejection.

Run the numerical C++ tests without ROS, from the package directory:

```bash
mkdir -p .build-check
g++ -O2 -std=c++17 -Iinclude test/test_metric_estimator.cpp \
  -o .build-check/test_metric_estimator $(pkg-config --cflags --libs opencv4)
.build-check/test_metric_estimator config/bottom_camera.yaml
```

Local numerical results: 66 checks passed; exact translation error about `9.1e-8 m/s`, pure rotation apparent speed about `3.7e-8 m/s`, noisy/outlier velocity RMSE about `0.0011 m/s`, and 30/30 noisy trials within the conservative covariance ellipsoid. These are synthetic numerical tests, not measured AUV accuracy or calibration validation. Full ROS compilation, the actual-node contract test and GUI execution require ROS 2; this editing environment does not have `ament_cmake`.
