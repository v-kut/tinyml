# Why three stages, and what each is worth

`tinyml-train` runs `regress -> ppo -> distill`. Only the middle stage is required.

## Cloning is a warm start, not a shortcut

A clone of the expert reproduces about three quarters of the variance in each action
channel per step, and closed still crashes on most layouts: per-step imitation at that
error does not survive a loop.

The binding constraint is capacity. Training loss tracks validation loss to within a few
percent, so nothing is memorized, and a much wider net on the same dataset fits steering
an order of magnitude better and drives several times as far. The small net cannot
represent pure pursuit, and PPO trains it to lap anyway, which is the point: it is free to
find a policy that fits in a couple of thousand parameters rather than one reproducing a
controller with a map.

What cloning buys is the start of PPO. From scratch the first part of the budget goes on
getting worse; warm-started, reward improves monotonically from the first evaluations.
Three things must travel:

- the `VecNormalize` statistics, since the trunk is a function of normalized observations
  and a fresh wrapper normalizes by its first batch's variance;
- the critic, fit in the reward units `VecNormalize` will produce, because a good actor
  beside a random critic gives advantages that say nothing;
- a smaller initial exploration std. SB3's default is std 1.0 across an action range of 2,
  which on top of a clone makes the first rollouts random.

The dataset is not the expert's own trajectory: the executed action carries
N(0, `noise_std`) while the label stays the expert's, so the states are ones a slightly
worse driver reaches and the labels are the way back (Laskey et al., DART).

## Distillation

`pi_arch` is sized for the optimizer, `student_arch` for flash, and the student ships. The
stage is skipped when the two match, since fitting a net into its own shape only clones
it. `pretrain.pt`, `ppo.pt` and `student.pt` keep each stage's result; `snapshot.pt` is
whichever ran last.

## The credit window is a duration

Per-step factors mean a different amount of time at every control rate, so `gamma` and
`gae_lambda` are configured as durations and converted at `dt`.

The window is 3 s, not the 1 s it was. The manoeuvre it must carry credit across is a
corner setup, and the racing line's swing from the outside to the apex and back measures
64-74 m, 1.4-1.6 s at racing speed. At 1 s the window was `1/(1 - gamma*lambda)` = 46
steps = 0.91 s, so the reward for making the apex decayed before credit reached the
decision to run wide, leaving the positioning to the critic. 3 s puts it at 116 steps. The
cost is the usual bias-for-variance trade, affordable because the critic is free.

## Actor and critic are sized independently

SB3 builds them as separate networks either way. The critic is discarded at export, so its
units are free on the device, and a value function regressing a 500-step return with a
crash cliff in it wants more of them than a 61 to 2 policy.

## Reward, and three ways an episode ends

Dense progress along the racing line, a penalty for leaving the track, a small
steering-rate penalty, and potential-based shaping on distance to the line. The shaping
telescopes and the potential is 0 at the terminal state, so it cannot change the optimal
policy (Ng et al., 1999).

Leaving the track terminates, the episode budget truncates, and covering under 1% of top
speed for `stall_seconds` truncates too. The last is economics: standing still pays zero,
so nothing in the reward ends the episode, and a parked car costs the full budget.
`episode/stall_rate` reports it.

## The live display and the `s` key

Each stage is one line updated in place. `s` finishes the current stage, which keeps what
it produced, publishes, scores and hands over, so a skipped PPO run still exports and exits 0. A keypress rather than a signal because Ctrl-\ goes to the whole process group, and
PPO's group is `n_envs` workers with no handler for it. `cbreak` leaves ISIG alone, and the
mechanism is a no-op when stdin is not a terminal.

## The trap in the two step flags

`--ppo-n-steps` is the rollout buffer per worker; `--total-steps` is the budget. SB3 fills
the buffer before every update and `collect_rollouts` ignores the remaining budget once
started, so a rollout wider than the budget collects `n_steps * n_envs` steps, updates
once, and publishes the policy the stage began with, while preallocating
`n_steps * n_envs * obs_dim` float32s. Neither the step count nor the reward curve looks
wrong alone, so `train_ppo` warns.
