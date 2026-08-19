# What the policy is allowed to see

`obs_dim` is the only statement of the layout, and every block is a width, not a flag: 0
drops it and narrows the network. At the defaults, 61 floats:

| block                       | flag                    | width   | what it is                                     |
| --------------------------- | ----------------------- | ------- | ---------------------------------------------- |
| `scan`                      |                         | 28      | LiDAR ranges over 240 deg, normalized          |
| sweep differences           | `--scan-history`        | 28      | `scan_t - scan_{t-1}`, then its own difference |
| `[vx, vy, yaw_rate, steer]` |                         | 4       | proprioception one LiDAR frame cannot carry    |
| past throttle               | `--throttle-history`    | 1       | the pedal that produced this state             |
| signed cross-track          | `--cross-track-history` | 0 (off) | offset from the racing line                    |

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

## Every block is scaled by its producer, and normalized again anyway

Each block leaves its owner in normalized units: ranges divided by the LiDAR's reach,
velocities by top speed, cross-track by the corridor half-width. That looks like enough,
which raises the obvious question of why `VecNormalize(norm_obs=True)` sits on top and why
its statistics then show up in `snapshot.pt` and get folded into layer 0 at export.

Hand scaling and whitening are not the same operation. Over 4000 expert steps at the
default layout, per-feature standard deviations run from 0.06 to 0.57, a nine-fold spread,
with a median of 0.12 and means as far off zero as 0.55. The units are bounded and
comparable; the distribution is neither centred nor unit-variance, so `obs_rms` is still
applying roughly an eight-fold per-feature gain.

Whether PPO needs that is a question for an experiment, not for a reading of the code, so
both arms were run: the shipping pipeline (clone, then PPO) against the same pipeline with
observation normalization deleted end to end, identity statistics in the regression stage,
`norm_obs=False` in both vec-envs, raw observations through `Snapshot.act`. Reward
normalization was left alone. Scores are `closed_loop` over 12 held-out layouts, clean
sensor.

| budget      | seeds | normalized   | raw         |
| ----------- | ----- | ------------ | ----------- |
| 400k steps  | 3     | 2519 +/- 644 | 634 +/- 173 |
| 1M steps    | 2     | 3859         | 413         |
| laps, at 1M | 2     | 1.19         | 0.16        |

The ranges never overlap: the worst normalized seed at 400k beats the best raw seed. Nor is
the raw arm merely slower. Over the last 100k steps of the 400k runs it gained 3.2 reward
per 1k steps against 19.5 normalized, and across a full 1M it never left the 80 to 660
band while the normalized arm reached 4541.

The interesting half is where there is no difference. Cloning scores 65 +/- 36 against
40 +/- 50 reward, but drives 0.08 laps either way: supervised regression onto hand-scaled inputs does
not care. The whole cost lands on PPO, which is consistent with the spread above leaving a
shared actor-critic trunk poorly conditioned at `--ppo-learning-rate 3e-5`. That mechanism
was not isolated; a learning-rate sweep on the raw arm is the experiment that would.
