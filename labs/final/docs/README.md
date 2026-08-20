# `docs/`

| path                          | what it is                          |
| ----------------------------- | ----------------------------------- |
| `findings/`                   | why the code is what it is          |
| `presentation/`               | slides                              |
| `PROPOSAL.md`, `proposal.pdf` | the original proposal, as submitted |
| `materials/`                  | background reading, third-party     |

## Materials

- Thomas, A. J., Petridis, M., Walters, S. D., Malekshahi Gheytassi, S., Morgan,
  R. E. _Two Hidden Layers are Usually Better than One._ EANN 2017, 279-290.
  [doi:10.1007/978-3-319-65172-9_24](https://doi.org/10.1007/978-3-319-65172-9_24)
  One and two hidden layers compared node by node over 1-64 nodes on ten
  function-approximation datasets; two won in nine. The evidence for not going deeper
  than two hidden layers in the deployed actor.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. _Proximal Policy
  Optimization Algorithms._ [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
  The algorithm behind the PPO stage, via Stable-Baselines3.
- Ng, A. Y., Harada, D., Russell, S. _Policy Invariance Under Reward
  Transformations: Theory and Application to Reward Shaping._ ICML 1999. The
  potential-based shaping theorem the cross-track term relies on, and why the potential
  must vanish at the terminal state (`ml/env.py`).
- Laskey, M., Lee, J., Fox, R., Dragan, A., Goldberg, K. _DART: Noise Injection for
  Robust Imitation Learning._ CoRL 2017,
  [arXiv:1703.09327](https://arxiv.org/abs/1703.09327). Noise on the executed action
  with the teacher's own label, which is how the cloning stage builds its dataset
  (`ml/regression/dataset.py`).
- Krishnamoorthi, R. _Quantizing Deep Convolutional Networks for Efficient
  Inference: A Whitepaper._ [arXiv:1806.08342](https://arxiv.org/abs/1806.08342)
  Per-output-channel int8 weight scales with per-tensor activation scales, the scheme
  `deploy/quantize.py` implements.
- `Choosing DNN Architectures for TinyML Tasks.pdf`, `TinyML Compression
Cookbook1.pdf`: course handouts.
