# MemGR00T-RMBench

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Project status](https://img.shields.io/badge/status-research%20preview-orange.svg)](docs/PROJECT_STATUS.md)

MemGR00T-RMBench is a research implementation of episode-local fast memory for
NVIDIA Isaac GR00T N1.7, with an auditable deployment path for the dual-arm
RMBench benchmark.

The project adds a lightweight **RoboTTT-style** adapter to selected action-DiT
layers. It is an engineering transfer of the fast-memory idea, not a claim of a
parameter-for-parameter reproduction of the original RoboTTT system.

> [!IMPORTANT]
> This is an independent derivative project maintained by `radiotoefx`. It is
> not an NVIDIA product and is not affiliated with or endorsed by NVIDIA.

## Why this project exists

Long-horizon manipulation policies repeatedly observe the same scene while
their actions change the world. MemGR00T explores whether a small recurrent
fast-weight state inside the action expert can use that ordered history without
fine-tuning the full vision-language backbone at every step.

The repository focuses on three properties:

- **real recurrent memory** — one committed memory update per policy query;
- **auditable deployment semantics** — explicit state/action identity, reset,
  freshness, retry, checkpoint, processor, and statistics metadata;
- **causal evaluation** — compare normal memory with reset-every-query and
  mid-episode reset while holding slow weights and policy randomness fixed.

## Architecture

```text
RMBench observation
        |
        v
GR00T vision-language backbone
        |
        v
action DiT layer(s)
        |
        +--> slow Q / K / V projections
        |         |
        |         v
        |    fast MLP update: K -> V
        |         |
        |         v
        +---- query with Q + learned residual gate
        |
        v
flow-matching action decode
        |
        v
canonical absolute dual-arm action chunk
```

The fast state is episode-local. Diffusion/flow denoising substeps reuse the
same state and do not count as additional observations. `reset()` clears memory
at an episode boundary; reseeding policy noise is a separate operation.

## Implemented

- Q/K/V projections and a two-layer GeLU fast model;
- inner K-to-V gradient update and learned residual gate;
- configurable insertion into selected action-DiT layers;
- state carry, detach, clone/restore, and per-layer diagnostics;
- ordered sequence training with configurable TBPTT boundaries;
- fail-closed sequence collation for mixed or unequal-length batches;
- independent policy RNG and episode-common initial flow noise;
- idempotent request retry and explicit episode reset over ZeroMQ;
- RMBench v4 absolute-action boundary with no second state addition;
- checkpoint/processor/statistics hashes and launch-manifest identity.

See [the mechanism document](docs/robottt_rmbench.md) for implementation detail.

## Current evidence and limits

| Area | Status |
| --- | --- |
| Fast-memory implementation | Source and unit-test complete |
| Ordered sequence/TBPTT path | Implemented; focused regression passing |
| Vanilla v4 H16 service contract | Full-model smoke verified |
| Historical RoboTTT pilot | Available as a prerelease checkpoint |
| Formal actual-state RoboTTT candidate | Not yet trained |
| Closed-loop RoboTTT improvement | Not yet demonstrated |
| Original-paper parity | Not claimed |

The published pilot was trained from a historical Vanilla-3k parent using the
older command-state RMBench data. It is useful for mechanism inspection, not as
evidence of task-level improvement. Read [Project Status](docs/PROJECT_STATUS.md)
and [Checkpoints](docs/CHECKPOINTS.md) before using it.

## RMBench boundary contract

The current deployment boundary is deliberately explicit:

```text
observation arm     = actual articulation qpos
observation gripper = physical normalized aperture
arm action          = absolute joint target
gripper action      = absolute target
formal action chunk = 16 steps
```

Internally, the checkpoint processor may represent arm actions relative to the
current state. `processor.decode_action()` restores absolute arm targets. The
server returns that decoded result directly and must not add state a second
time.

## Installation

This repository retains the upstream `gr00t` Python package/distribution name
for checkpoint and API compatibility; it is an independent derivative and is
not the NVIDIA package release. GR00T N1.7 currently targets Python 3.12.

The complete installation, standard GR00T inference, Policy API, server,
fine-tuning, and verification flow is in [`docs/USAGE.md`](docs/USAGE.md).
Install Git LFS before cloning for a normal demo-data checkout; CI and
source-only users can use the documented `GIT_LFS_SKIP_SMUDGE=1` flow.

The upstream platform-specific CUDA and edge-device instructions remain
available in [`scripts/deployment/README.md`](scripts/deployment/README.md).

## Focused tests

```bash
uv run pytest -q \
  tests/gr00t/model/test_robottt.py \
  tests/gr00t/model/test_robottt_sequence.py \
  tests/gr00t/policy/test_gr00t_policy.py \
  tests/gr00t/policy/test_policy_service.py \
  tests/gr00t/policy/test_rmbench_adapter.py
```

These tests cover fast-state updates, sequence/TBPTT lifecycle, reset, noise,
freshness, retry idempotency, absolute action semantics, and finite guards.

For the complete supported usage path, including upstream-compatible GR00T
commands, see [`docs/USAGE.md`](docs/USAGE.md).

## Serving RMBench

The server entry point is the existing GR00T policy service with additional
identity and memory diagnostics:

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --processor-path /path/to/checkpoint \
  --modality-config-path examples/RMBench/rmbench_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555
```

Production launches should use
[`scripts/ppu/run_gr00t_rmbench_policy_server.sh`](scripts/ppu/run_gr00t_rmbench_policy_server.sh)
to require explicit device, seed, paths, semantics, and manifest output.

## Repository map

- `gr00t/model/modules/robottt.py` — fast-memory implementation;
- `gr00t/model/gr00t_n1d7/` — N1.7 action-head and sequence integration;
- `gr00t/policy/` — reset/noise/service and RMBench boundary protocol;
- `examples/RMBench/` — formal H16 modality configuration;
- `tests/gr00t/model/` — mechanism and sequence regressions;
- `tests/gr00t/policy/` — wire protocol and boundary regressions;
- `docs/` — design, evidence, checkpoints, and limitations.

## Roadmap

1. Freeze an actual-qpos + physical-gripper RMBench dataset conversion.
2. Train and select a closed-loop-capable actual-state Vanilla parent.
3. Fork Vanilla and RoboTTT from identical slow weights.
4. Run paired normal/reset/mid-reset evaluations with common seeds and noise.
5. Publish per-query memory traces and task-level confidence intervals.

## Upstream and attribution

This project is based on
[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) and preserves its
Apache-2.0 license and attribution. See [NOTICE](NOTICE) for the derivative-work
statement.

The memory mechanism is inspired by RoboTTT. Users should cite the original
GR00T and RoboTTT papers when reporting results based on this repository. The
historical pilot in this repository does not establish reproduction of the
paper's reported results.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
