# Front camera

This component owns front camera capture, stereo calibration, and color rectification. It may contain several ROS nodes. ROS package: `hornet_front_camera`.

## Files

- `hornet_front_camera/calibration.py`: `front_calibration` runtime node; splits and rectifies a side-by-side image.
- `hornet_front_camera/calibrate.py`: `calibrate_stereo` offline checkerboard command; writes left/right calibration YAML.
- `hornet_front_camera/models.py`: saved camera-model loading and stereo geometry validation; also used by depth estimation.
- `config/examples/{left,right}.yaml`: synthetic geometry examples, disabled by default.
- `config/measured/{left,right}.yaml`: saved tutorial stereo calibration, 320×240 per lens. See [profile details and launch commands](config/measured/README.md).
- `config/rectification.yaml`, `launch/rectify.launch.py`: runtime configuration and launch.
- Add camera-driver configuration, launch, and any `.py` acquisition adapter in this component. Prefer an existing Jazzy driver such as `usb_cam`.

## Interfaces

Capture publishes `/front/image_raw` (`sensor_msgs/msg/Image`, sensor-data QoS), containing equal-width horizontal stereo halves with no separator. Preserve acquisition timestamps. Record the physical camera identity, capture format, focus, resolution, FPS, exposure synchronization, and half ordering.

`front_calibration` reads saved left/right YAML once and publishes `/front/left/image_rect` and `/front/right/image_rect` as **BGR8 color**. Outputs preserve the input stamp and use the saved rectified optical frame IDs. It rejects inconsistent dimensions and has no focal-length/baseline or CameraInfo publishers. `swap_input_halves` handles capture ordering, not an incorrectly labelled calibration.

## Offline calibration

Capture varied checkerboard poses at the deployment capture mode and fixed focus. Save synchronized per-lens left/right images with identical filenames in two directories. Use actual inner-corner counts and measured square size in metres.

```bash
ros2 run hornet_front_camera calibrate_stereo \
  --left-dir ~/calibration_images/left --right-dir ~/calibration_images/right \
  --board-cols 9 --board-rows 6 --square-size-m 0.025 \
  --output-dir ~/hornet_calibration/front_run_01
```

The board settings are examples. The tool estimates intrinsics, stereo extrinsics and rectification; only horizontal pinhole stereo is supported. It requires at least 12 accepted views by default, rejects fits exceeding its RMS gate, and never overwrites an output directory. Validate held-out alignment, physical baseline and measured ranges in the actual housing/medium. Store camera/mode/board provenance with the YAMLs.

## Runtime

To reuse the calibration saved in the ROS 2 Jazzy tutorial, follow the [measured profile instructions](config/measured/README.md). It requires the same physical stereo rig, focus and capture mode, with 640×240 combined input. The launch defaults still point to disabled synthetic examples, so select the measured pair explicitly.

To use a newly generated calibration instead:

```bash
ros2 launch hornet_front_camera rectify.launch.py \
  left_yaml:="$HOME/hornet_calibration/front_run_01/left.yaml" \
  right_yaml:="$HOME/hornet_calibration/front_run_01/right.yaml"
```

The camera driver must publish separately. A 320×240 per-lens calibration requires 640×240 combined input; do not resize a different capture mode to conceal mismatched geometry. Depth and detector consumers load the same calibration ID. Coordinate raw/rectified camera mounting TF with the localization component. See [interfaces](../docs/interfaces.md) and [frames](../docs/frames.md).
