# What the policy is allowed to see

`obs_dim` is the only statement of the layout, and every block is a width, not a flag: 0
drops it and narrows the network. At the defaults, 61 floats:

| block | flag | width | what it is |
| --- | --- | --- | --- |
| `scan` | | 28 | LiDAR ranges over 240 deg, normalized |
| sweep differences | `--scan-history` | 28 | `scan_t - scan_{t-1}`, then its own difference |
| `[vx, vy, yaw_rate, steer]` | | 4 | proprioception one LiDAR frame cannot carry |
| past throttle | `--throttle-history` | 1 | the pedal that produced this state |
| signed cross-track | `--cross-track-history` | 0 (off) | offset from the racing line |

## Differenced sweeps are not for closure rate

Against a static wall, closure rate is derivable from the scan and `[vx, vy, yaw_rate]`,
so a second frame adds no geometry. It adds a second look: `dropout_prob` garbles
readings, so a sizeable fraction of frames arrive with a dead beam, and the centre beams,
the only ones with reach, are as likely to be it as any other. The previous sweep lets the
policy ride over a dropout instead of steering at it.

On a clean detector that benefit cannot exist, so the block is scored in both conditions:
`evaluation_variant(clean_sensor=...)` changes the detector and nothing else, `train.score`
logs both passes per stage, and `tinyml-build --sensor noisy` drives the deployment table
behind the training sensor.

Differences rather than raw sweeps, since two near-identical raw blocks would share one
int8 absmax and spend their range on the part that did not move.

Open: the `--scan-history 1` vs `2` A/B on the noisy score decides whether the extra 28
inputs stay, and has not been run.

## Past throttle

`steer` is the achieved, slew-limited angle, so the previous steering command is already
observable through the lag. Nothing else reports that the car is braking.

## Signed cross-track is privileged state, off by default

It says where the car is relative to the racing line, which the simulator knows because it
solved for it and a real car would need a map for. It closes an otherwise open loop, since
the shaping term is a function of exactly this quantity. Off anyway: the deployed policy
should read only what a range finder and the car's own sensors supply.

Signed because the potential needs the magnitude and a policy needs the sign to know which
way to steer back. A history rather than one scalar because two readings give the rate, and
the spawn's own offset seeds it.

## The lookahead-curvature block was removed

Sampling corridor curvature at six distances ahead moved a linear read-off of signed
cross-track only from R^2 0.40 to 0.44 (twelve samples: 0.46), while the base observation
already explains 99.1% of the expert's steering linearly. The block cost six int8 inputs
to say so.
