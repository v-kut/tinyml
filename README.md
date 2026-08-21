# tinyml

This is a repo for EE446 TinyML course.

Coursework lives in [`labs/`](labs) and [`homework/`](homework). The final project,
[`labs/final/`](labs/final), is a standalone project with its own environment, license
and [setup instructions](labs/final/README.md) — nothing below applies to it.

## Setup

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — provisions the pinned
  CPython, so no system Python is touched.
- `git` — only if you want the Arduino TFLM examples library cloned for you.

The script targets Arch Linux, but nothing in it is distro-specific beyond the package
hints in its error messages.

### Course environment

```bash
bash setup_env.sh
```

This creates `tinyml-arduino/` next to the script: CPython 3.10.14 with TensorFlow 2.14.1,
Keras 2.14.0, TF-MOT 0.8.0 and `numpy<2` — the versions the course tooling expects. It also
registers a Jupyter kernel named `tinyml-arduino`, writes
`tinyml-arduino/tinyml_env_smoke_test.py`, and runs it.

The smoke test trains a small classifier and exercises every compression path used in the
labs — FP32 / dynamic-range / full-INT8 conversion, magnitude pruning, QAT, distillation,
and C header export. It prints `ALL SMOKE TESTS PASSED` on success.

Activate it when working on a lab:

```bash
source tinyml-arduino/bin/activate
```

TensorFlow is installed as the plain CPU wheel. It will still use a GPU if a matching
system CUDA runtime happens to be present; the smoke test prints what it detects.

### Options

Both positional arguments are optional — `bash setup_env.sh <env-name> <base-dir>`
relocates the environment.

| variable               | default                | effect                                            |
| ---------------------- | ---------------------- | ------------------------------------------------- |
| `RUN_SMOKE_TEST`       | `1`                    | Run the smoke test at the end                     |
| `REINSTALL`            | `0`                    | Delete and recreate the environment if it exists  |
| `INSTALL_ARDUINO_TFLM` | `0`                    | Clone the Arduino TFLM examples library           |
| `TFLM_REPO_URL`        | official upstream      | Repo to clone for the TFLM library                |
| `TFLM_REPO_REF`        | default branch         | Branch/tag/commit to check out                    |
| `PY_VERSION`           | `3.10.14`              | Pinned CPython version                            |
| `TF_VERSION`           | `2.14.1`               | Pinned TensorFlow version                         |
| `KERAS_VERSION`        | `2.14.0`               | Pinned Keras version, must match `TF_VERSION`     |
| `TFMOT_VERSION`        | `0.8.0`                | Pinned TF-MOT version, must match `TF_VERSION`    |

`INSTALL_ARDUINO_TFLM=1` clones into `~/Arduino/libraries/Arduino_TensorFlowLite` and skips
the clone if that directory already exists. Upstream
[`tflite-micro-arduino-examples`](https://github.com/tensorflow/tflite-micro-arduino-examples)
is archived and read-only.

### Notebooks

Most labs are [Marimo](https://marimo.io/) notebooks under `<lab>/marimo/`, served from the
activated environment:

```bash
marimo edit labs/lab7/marimo
```

### Per-lab tooling

- **Arduino sketches** — flash the `.ino` under a lab's `Hardware/` or `submission/`
  directory with the Arduino IDE or `arduino-cli`, targeting `arduino:mbed_nano`.
- **Lab 4** uses the Edge Impulse CLI instead of a local environment:
  `cd labs/lab4 && bun install`.

## License

This repo strives to be [REUSE](https://reuse.software/) compliant.

Generally:

Documentation is licensed under [CC-BY-NC-4.0](LICENSES/CC-BY-NC-4.0.txt)  
Code is licensed under [MIT](LICENSES/MIT.txt)  
Config files, datasets and generated model artifacts are under [CC0-1.0](LICENSES/CC0-1.0.txt)
