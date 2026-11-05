# Saved tutorial stereo calibration

`left.yaml` and `right.yaml` are the measured stereo calibration recovered from the Ubuntu VMware machine used in **Explore ROS 2 Jazzy tutorials**. Both were checked byte-for-byte against the YAML members of the same saved `~/stereo_calibration/calibrationdata.tar.gz` archive before import.

| Property | Value |
|---|---|
| Result file timestamp | 2026-10-30 08:56:03 UTC |
| Saved archive timestamp | 2026-10-30 09:02:01 UTC |
| Per-lens image size | 320×240 |
| Combined side-by-side input | 640×240 |
| Distortion model | `plumb_bob` |
| Checkerboard inner corners | 8×6 |
| Entered square size | 0.025 m |
| Rectified focal length | 239.38129 px |
| Stereo baseline from right P | Approximately 0.11918672 m |
| Shared calibration ID | `tutorial_stereo_2026_10_30_ffd6ed5c_916eae4e` |
| Left rectified frame | `front_left_rect_optical_frame` |
| Right rectified frame | `front_right_rect_optical_frame` |

Board settings come from the calibration command in the VM's shell history. The saved results retain all original camera names, dimensions and K/D/R/P values. Repository metadata was appended to declare metre units, the shared calibration ID, rectified frame IDs and source hashes. The hashes refer to the original YAML bytes before that metadata was appended:

| Source | SHA-256 |
|---|---|
| `left.yaml` | `ffd6ed5cc733c187b63b0c1acef9a46bedf6ab0b0aad42caa74e2b181addc6cd` |
| `right.yaml` | `916eae4e5129716828a1bc3b0af18a38e3373efb971c2a0941a0a5e722dc111a` |

The archive, calibration images and unrelated VM files are not included in this repository. Hardware serial numbers, focus, housing/medium, capture FPS and measured range accuracy were not recorded in these YAMLs. Confirm that the rig and camera mode match before reusing this profile. The baseline is a calibration estimate, not an independently measured distance. Changing the computer alone does not require recalibration; changing camera geometry, focus, housing/medium or image formation can require it.

## Use the saved pair

After building and sourcing your ROS 2 workspace, run the front rectifier:

```bash
front_share="$(ros2 pkg prefix --share hornet_front_camera)"
ros2 launch hornet_front_camera rectify.launch.py \
  left_yaml:="$front_share/config/measured/left.yaml" \
  right_yaml:="$front_share/config/measured/right.yaml"
```

In another terminal, source the same workspace and start depth estimation:

```bash
front_share="$(ros2 pkg prefix --share hornet_front_camera)"
ros2 launch hornet_depth depth.launch.py \
  left_yaml:="$front_share/config/measured/left.yaml" \
  right_yaml:="$front_share/config/measured/right.yaml"
```

The camera driver must publish 640×240 side-by-side images on `/front/image_raw`, or pass its actual topic as the rectifier's `input_topic` launch argument. These commands do not start the driver, detection producers or localization. Depth requires matched `StereoObservation` messages with the calibration ID above and timestamped TF into `map`. Camera mounting and raw-to-rectified TF must be supplied by integration; R is nonidentity in this pair, so follow [the frame conventions](../../../docs/frames.md).

The measured pair is installed with `hornet_front_camera`. Launch defaults remain the synthetic examples, which are disabled unless explicitly allowed; the arguments above select this measured profile. Use the same YAMLs throughout the pipeline and check rectified alignment and known physical ranges before relying on the resulting positions. The bottom camera has its own calibration and cannot use this stereo pair.
