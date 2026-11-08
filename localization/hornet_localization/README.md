# Hornet localization ROS package

This package is the initial ROS 2 Jazzy integration point for Hornet XII
localization.  It is a development scaffold, not a validated real-robot
localization solution.

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
| Input | `/sensors/magnetic_field` | `sensor_msgs/msg/MagneticField` |
| Output | `/localization/odometry` | `nav_msgs/msg/Odometry` |

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

## Current limitations

- The estimator method is intentionally not implemented yet.
- The node does not publish fake zero odometry while its estimate is invalid.
- IMU-only position will drift and cannot provide stable absolute X/Y.
- Real covariances, sensor mounting transforms and dropout behavior remain to
  be agreed and tested.

## Build

From a ROS workspace containing this repository:

```bash
colcon build --packages-select hornet_localization
source install/setup.bash
```

Run the node with:

```bash
ros2 run hornet_localization imu_localization_node
```

or the launch file with:

```bash
ros2 launch hornet_localization localization.launch.py
```
