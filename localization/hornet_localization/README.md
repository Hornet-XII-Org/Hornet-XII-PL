# Hornet localization ROS package

This package is the initial ROS 2 Jazzy integration point for Hornet XII
localization.  It provides the first working IMU-to-odometry pipeline, but it
is not yet a validated real-robot localization solution.

## Parallel ownership

Nix owns:

- `imu_localization_node.py`
- ROS subscriptions and parameters
- `/localization/odometry` publication
- dynamic `odom -> base_link` TF publication
- conversion of map-axis estimator velocity into `base_link` axes for
  `Odometry.twist`

Dao Quang Linh owns:

- `inertial_estimator.py`
- Madgwick configuration and launch integration
- gravity removal and acceleration integration
- estimator tests and drift characterization

Avoid editing the other owner's files without coordinating first.

## Proposed interfaces

These defaults follow the repository contracts but still require integration
confirmation with the sensor-bridge and consumer owners.

| Direction | Topic | Type |
| --- | --- | --- |
| Input | `/sensors/imu` | `sensor_msgs/msg/Imu` |
| Reserved input | `/sensors/magnetic_field` | `sensor_msgs/msg/MagneticField` |
| Output | `/localization/odometry` | `nav_msgs/msg/Odometry` |

The launch chain passes `/sensors/imu` through Madgwick and the yaw aligner
before the localization node receives it.  Magnetometer fusion is currently
disabled (`use_mag: false`); the magnetic-field topic is reserved for later
hardware validation.

All calculations use acquisition timestamps from input message headers.  Do
not replace them with callback-completion time.

The initial TF scope is `map -> odom -> base_link`.  The launch integration may
provide an identity `map -> odom` while the node owns the changing
`odom -> base_link` transform.  Camera and sensor mounting transforms require
measured extrinsics and are outside this initial scaffold.

## Estimator contract

The estimator exposes:

```python
position_map, velocity_map = estimator.update(
    timestamp_sec,
    acceleration_body_mps2,
    orientation_map_from_base_xyzw,
)
```

Both returned vectors are expressed in map axes.  Before publishing
`Odometry.twist.twist.linear`, the node must rotate `velocity_map` by the
inverse orientation quaternion so the twist is expressed in `base_link` axes.

## Verification

Build and automated tests:

```bash
colcon build --packages-select hornet_localization
source install/setup.bash
colcon test --packages-select hornet_localization
colcon test-result --verbose
```

The automated suite covers estimator integration, rejected timestamps and
invalid values, quaternion/vector rotation, simulator axis conversion, and
yaw alignment.

### Stationary software smoke test

This test checks the complete ROS topic and TF wiring without claiming real
sensor accuracy.  Open three sourced terminals.

Terminal 1 — start the localization chain:

```bash
ros2 launch hornet_localization localization.launch.py
```

Terminal 2 — publish a level, stationary, synthetic IMU at 20 Hz:

```bash
ros2 topic pub --rate 20 /sensors/imu sensor_msgs/msg/Imu \
  "{header: auto, angular_velocity: {x: 0.0, y: 0.0, z: 0.0}, linear_acceleration: {x: 0.0, y: 0.0, z: 9.81}}"
```

Terminal 3 — confirm that odometry and the dynamic transform are available:

```bash
ros2 topic echo --once /localization/odometry
ros2 run tf2_ros tf2_echo odom base_link
```

For the stationary message above, position and velocity should remain close
to zero.  Stop each long-running command with Ctrl+C.

## Current limitations

- The node does not publish fake zero odometry while its estimate is invalid.
- IMU-only position will drift and cannot provide stable absolute X/Y.
- The angular-velocity output currently assumes the incoming IMU axes are
  aligned with `base_link`.
- Magnetometer fusion is disabled until its calibration and mounting are
  confirmed.
- Real covariances, sensor mounting transforms, dropout behavior, and measured
  stationary/movement drift remain to be agreed and tested on hardware.

## Run

Run only the final interface node with:

```bash
ros2 run hornet_localization imu_localization_node
```

Run the complete chain with:

```bash
ros2 launch hornet_localization localization.launch.py
```
