# T870 MCU feedback contract

Verified against T870_MCU commit `de026ddcd981e5c0ceb7305e664dfb94f56714a4`
and its effective `config/t870_mcu.yaml`.

| Topic | ROS type | Unit/sign | Validity | Nominal publication |
|---|---|---|---|---|
| `/mcu/encoder` | `std_msgs/msg/Int32` | Signed cumulative v37 odometer count; forward adds and reverse subtracts | Malformed or implausible jumps are not published | One message per accepted `STATUS`, normally polled at 5 Hz |
| `/mcu/distance_m` | `std_msgs/msg/Float32` | Nonnegative cumulative wheel-roll distance in metres; reverse also increases the total | Published only when `counts_per_meter > 0` and an encoder interval can be calculated | One message per calculated `STATUS`, normally 5 Hz |
| `/mcu/speed_mps` | `std_msgs/msg/Float32` | Signed metres per second | Same calculation gate; `/mcu/speed_valid` states whether `counts_per_meter` is configured | One message per calculated `STATUS`, normally 5 Hz |
| `/mcu/speed_valid` | `std_msgs/msg/Bool` | `true` means `counts_per_meter > 0` | This flag alone does not prove freshness; consumers also enforce a 0.5 s timeout | One message per accepted encoder `STATUS`, normally 5 Hz |
| `/odom` | `nav_msgs/msg/Odometry` by external contract | Not implemented by this MCU repository | Not verifiable from this repository | Not verifiable from this repository |

The MCU bridge's own reference odometry is effectively remapped to
`/mcu/odom` (`nav_msgs/msg/Odometry`) with TF publication disabled. The Python
source has a stale `/odom` default, but the shipped launch loads the YAML, so
the effective deployed topic is `/mcu/odom`.

`camera_ws` consumes the MCU scalar feedback directly. It does not derive
distance from `/odom`, derive speed from RPM/encoder deltas, or publish a wheel
odometry estimate. Validation launches do not start a serial bridge or vehicle
actuation node.
