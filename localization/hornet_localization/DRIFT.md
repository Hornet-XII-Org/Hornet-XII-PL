# IMU drift notes and node interface

**All results in this file are SYNTHETIC.** They come from fake data, not from
our sensor, the simulator or the robot.

**IMU-only position drifts and cannot give stable absolute X/Y.** Every small
acceleration error is added up twice, first into velocity and then into
position, so the error keeps growing. Stable X/Y needs another source of
position information.

## Drift results (SYNTHETIC)

### Assumptions

**These values are assumptions, not measurements from our sensor.** The real
IMU's noise, bias and orientation error, and the real gravity value, are
unknown.

| Assumed error | Value | How it is applied |
| --- | --- | --- |
| Noise | 0.02 m/s^2 (one sigma) | Random, added to X, Y and Z of every reading |
| Bias | 0.01 m/s^2 | Constant, added to body X |
| Tilt error | 1 degree | Constant pitch error in the orientation given to the estimator |
| Gravity error | 0.01 and 0.03 m/s^2 | The still reading on Z is 9.80 or 9.78, but the estimator subtracts 9.81 |

### Method

- The robot is level and still at the origin, so the true position is always
  (0, 0, 0).
- Data rate is 100 Hz. Each case is one 60 s run, read at 10, 30 and 60 s.
- The seed is fixed at 2026. Noise comes from Python's
  `random.Random(2026).gauss(0, 0.02)`, drawn in X, Y, Z order for each sample.
- Each sample goes directly into `InertialEstimator.update()`. Madgwick is not
  used in this table.
- Position error is the straight-line distance from the true position.

### Results (SYNTHETIC)

| Case | 10 s | 30 s | 60 s |
| --- | ---: | ---: | ---: |
| 1. Noise only | 0.066 m | 0.166 m | 0.314 m |
| 2. Bias only | 0.500 m | 4.500 m | 18.000 m |
| 3. 1 degree tilt error only | 8.561 m | 77.047 m | 308.186 m |
| 4. Noise, bias and tilt together | 9.075 m | 81.599 m | 326.222 m |
| 5. Gravity error 0.01 m/s^2 only (9.80 vs 9.81) | 0.500 m | 4.500 m | 18.000 m |
| 6. Gravity error 0.03 m/s^2 only (9.78 vs 9.81) | 1.500 m | 13.500 m | 54.000 m |

### What the table shows

- A constant error `a` gives a position error of about `0.5 * a * t^2`. Three
  times the time gives nine times the error. Cases 2, 5 and 6 match this
  exactly.
- In cases 5 and 6 all the error is on Z, and the estimate moves downward,
  because the estimator subtracts more gravity than the sensor measured.
- Tilt error matters most. A 1 degree error turns part of gravity into false
  sideways acceleration of `9.81 * sin(1 degree) = 0.171 m/s^2`, which is 17
  times the assumed bias.
- Case 4 is a bad case: bias and tilt both push along +X here, so they add. In
  a different direction they could partly cancel.
- Case 1 is one random run. Another seed gives different numbers.
- The table holds orientation constant. It does not include gyro drift, or any
  change in Madgwick's orientation.

### Effect of Madgwick (SYNTHETIC, live chain)

Fake still data was sent through Madgwick and the yaw aligner into the
estimator for 10 s.

| Fake acceleration input | Largest position error in 10 s |
| --- | ---: |
| Exactly level, (0, 0, 9.81) | 0.000 m |
| Tilted about 1.17 degrees, same size 9.81 | 0.056 m and 0.126 m (two runs) |

The exactly level result is a special case, because the orientation stays
exactly right and never changes. With the small tilt, the orientation moved by
about 0.001 (quaternion length) at every sample even though the input never
changed, and roll moved between 1.03 and 1.29 degrees. That step size equals
Madgwick's gain times the sample time (0.1 x 0.01 s), so the filter seems to
take a fixed-size step even when its error is tiny. The size and direction of
the drift changed between the two runs.

## Node interface

`README.md` describes the full chain. This section gives the interface details
for the two extra nodes and the estimator. `CONTRIBUTING.md` still requires
producer and consumer review of interface changes.

### `yaw_aligner_node`

- **Input:** `/localization/imu/filtered`, `sensor_msgs/msg/Imu`.
- **Output:** `/localization/imu/aligned`, `sensor_msgs/msg/Imu`.
- **Frames:** `header.frame_id`, acceleration and gyro are copied unchanged.
  The orientation is the robot's orientation relative to `map`, with the yaw of
  the first message removed.
- **Timestamps:** `header.stamp` is copied unchanged.
- **QoS:** reliable, volatile, keep last 10 on both topics.
- **Invalid data:** the starting yaw is saved only from a finite, non-zero
  quaternion (length at least 1e-6). Until one arrives, messages pass through
  unchanged, so a bad first message cannot spoil the starting yaw. After the
  yaw is saved there are no more checks: a zero quaternion stays zero, and NaN
  stays NaN. `imu_localization_node` then rejects these messages before they
  reach the estimator.

### `sim_imu_adapter_node` (simulator only)

- **Input:** `/model/bluerov2/imu`, `sensor_msgs/msg/Imu`.
- **Output:** `/sensors/imu`, `sensor_msgs/msg/Imu`.
- **Frames:** applies `(x, -y, -z)` to acceleration and gyro, and sets
  `header.frame_id` to `base_link`. `orientation_covariance[0]` is set to -1
  (no orientation available).
- **Timestamps:** `header.stamp` is copied unchanged.
- **QoS:** input is best effort, volatile, keep last 5, which can receive from
  reliable and best-effort publishers. Output is reliable, volatile, keep last
  10.
- **Invalid data:** there are no checks. Bad values are passed on.

### `InertialEstimator`

This is a Python class, not a ROS node.

- **Input:** `update(timestamp_sec, acceleration_body_mps2,
  orientation_map_from_base_xyzw)`.
- **Output:** `(position_map_m, velocity_map_mps)` as tuples, or `None`.
- **Frames:** acceleration is the raw reading in `base_link` axes, including
  gravity (level and still reads `(0, 0, +9.81)`). The quaternion is
  `map` from `base_link`, in x, y, z, w order. Both outputs are in `map` axes.
- **Timestamps:** the caller must pass the acquisition time from the message
  header, in seconds, all from the same clock.
- **QoS:** none, because this class does not use ROS. The calling node chooses.
- **Invalid data:** returns `None` and keeps its state for NaN or infinite
  input, a near-zero quaternion, a repeated timestamp, or a backward step of
  `max_dt_sec` (default 0.5 s) or less. For a jump forward or backward of more
  than `max_dt_sec`, it returns `None`, does not integrate across the jump, and
  restarts its timing from that sample.

### Map origin

`docs/frames.md` asks for the origin, heading and depth start to be documented.
For now:

- **Origin:** `map` starts where the robot is at the first sample. It is not a
  pool origin, so it can be different on every run.
- **Heading:** `map` X is the robot's heading at the first filtered message.
  There is no compass, so nothing corrects yaw later.
- **Vertical:** Z is up. Z = 0 is the starting height, not the water surface or
  the pool floor.

## Known limits

- **Gravity is fixed at 9.81.** `imu_localization_node` calls
  `InertialEstimator()` with no arguments, so the value cannot be changed at
  launch. Cases 5 and 6 above show the effect of a small difference.
- **After a data gap or clock jump, the old position and velocity are kept.**
  If the robot's speed changed during the gap, the old velocity is wrong, and
  it keeps being added to the position.
- **`Odometry` covariance is still all zeros.** Zero means "no uncertainty",
  which is not true for this estimate.
- **CPU load at 1000 Hz is not measured.**

## Still untested

Run so far, all with fake IMU data: unit tests, Madgwick through
`/localization/imu/aligned`, `map -> odom`, the simulator adapter on its own,
and starting all nodes with `use_sim:=true`.

Nix's `imu_callback` and its `None` handling now exist. `README.md` records a
stationary SYNTHETIC test of the full chain: `/localization/odometry` and the
TF lookups reported zero.

**Stationary chain test, repeated on 2026-11-09 (SYNTHETIC).** With
`ros2 launch hornet_localization localization.launch.py` running, the README's
level, still message was published on `/sensors/imu` at 20 Hz (237 messages,
about 12 s). All four nodes started with no errors. Two messages read from
`/localization/odometry`, about 5 s and 11 s after the first IMU message, both
had position and linear velocity of exactly 0.0. `tf2_echo odom base_link`
reported zero translation and the identity rotation. This only checks the
wiring: the input is perfectly level and has no noise, so zero is the expected
result. It says nothing about accuracy.

- Moving data through the full chain to `/localization/odometry` and
  `odom -> base_link`.
- Non-zero velocity and twist conversion through the running node. Only the
  maths has a unit test.
- A `None` result going through the running node.
- Any real simulator data. The simulator has not been run with this package.
- The simulator's IMU pose, QoS and sign of its readings.
- `use_sim:=true` with `/clock` or moving data. Only node startup was checked.
- Whether Python keeps up with the simulator's 1000 Hz IMU.
- Any real IMU. `sensor_bridge` has no code yet, so real `/sensors/imu` does
  not exist. Real axes, noise, bias and gravity convention are unknown.
- Real mounting between the IMU and `base_link`.
- Position estimates while the robot moves through the live chain. The live
  turn test checked yaw only.
- Roll or pitch motion through the live chain.
- Gyro bias and long-term yaw drift, with no compass.
- Data gaps, dropped messages and bag clock jumps through the live chain.
- CPU load and delay on the robot's computer.
