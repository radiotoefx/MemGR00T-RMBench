# RoboTTT mechanism on GR00T N1.7

> Historical mechanism document. For current runtime semantics, admissibility
> gates, and the active plan, use
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md). In particular, do
> not treat the old `lerobot_v2.1` dataset or Vanilla-3k examples below as formal
> actual-state candidates.

This branch implements a method-transfer version of RoboTTT for the public
GR00T N1.7 checkpoint. RMBench remains a separate rollout process and talks to
GR00T through the existing Policy API.

## What is implemented

- KVB test-time training: slow Q/K/V projections, a two-layer GeLU fast MLP,
  one inner MSE update from K to V, then a query with Q.
- Episode-local fast weights initialized from learned `W0`.
- A learned vector gate, initialized near zero, merges the TTT output into the
  action DiT residual stream.
- TTT blocks can be added to selected action-DiT layers without changing a
  vanilla checkpoint's stored tensors.
- In rollout, fast weights update once per environment observation. Diffusion
  denoising substeps reuse the updated state and do not count as new memories.
- Policy `reset()` clears fast weights. It must be called at every episode
  boundary.
- Sequence training uses ordered, fixed-length windows from a single LeRobot
  episode. It samples independent flow noise/time at every timestep, carries
  fast weights through the window, and detaches them at configurable TBPTT
  boundaries without resetting their values.
- Runtime diagnostics expose the memory step, update norm, and gate magnitude.

The public N1.7 checkpoint has 32 action-DiT blocks, whereas the RoboTTT paper
describes an internal 16-block GR00T action expert. The default here adds small
TTT modules to the final two blocks. This is deliberately an engineering
starting point, not a claim of parameter-for-parameter reproduction. The
current fast model is single-head and does not yet implement the paper's RoPE
variant. Set `ttt_num_register_tokens=16` only for a model trained with those
tokens; register tokens alter the base attention path even if the TTT gate is
zero.

## Environment

The repository does not assume a machine-specific workspace, model directory,
dataset mount, or device number. By default the activation helper uses the
repository `.venv` and `.cache` directories. Override `GR00T_ENV`,
`GR00T_CACHE_DIR`, `GR00T_MODEL_PATH`, and `GR00T_PPU_DEVICE` for a managed
installation. The helper unsets `HF_ENDPOINT` so downloads use the official
Hugging Face endpoint:

```bash
export GR00T_PPU_DEVICE=0  # replace with an authorized physical device id
source scripts/ppu/activate_gr00t.sh
```

## Verified focused tests

```bash
export GR00T_PPU_DEVICE=0  # replace with an authorized physical device id
source scripts/ppu/activate_gr00t.sh
python -m pytest -q \
  tests/gr00t/model/test_robottt.py \
  tests/gr00t/model/test_robottt_sequence.py \
  tests/gr00t/policy/test_policy_service.py \
  tests/gr00t/policy/test_rmbench_adapter.py
```

The maintained PPU environment also supports full-model smoke testing, but
those runs require local model/data paths and are not exposed as portable
repository commands.

## Continuous-sequence fine-tuning

Start with one or two TTT blocks and a short context. With `global_batch_size=1`,
one batch item is one whole sequence, not one frame.

```bash
export GR00T_PPU_DEVICE=0  # replace with an authorized physical device id
source scripts/ppu/activate_gr00t.sh

python gr00t/experiment/launch_finetune.py \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-path /path/to/rmbench-lerobot-dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/RMBench/rmbench_config.py \
  --output-dir /path/to/output/rmbench-ttt-l32 \
  --num-gpus 1 \
  --global-batch-size 1 \
  --use-ttt \
  --tune-ttt \
  --no-tune-projector \
  --no-tune-diffusion-model \
  --ttt-num-layers 2 \
  --ttt-dim 256 \
  --ttt-hidden-dim 1024 \
  --ttt-gate-init 0.001 \
  --ttt-num-register-tokens 0 \
  --ttt-sequence-training \
  --ttt-sequence-length 32 \
  --ttt-sequence-stride 32 \
  --ttt-tbptt-segment-length 8
```

Increase context only after the loss and update norm remain finite at the
previous length. A practical progression is 32, 128, then 512. For the first
stage keep the VLM and vanilla action path frozen; later runs can enable
`--tune-diffusion-model`.

The ordered dataset loader never crosses episode boundaries and drops an
episode tail shorter than a full window. Window order is randomized for SGD,
but timestep order inside every window is preserved. Memory resets at the start
of each window; TBPTT segments inside a window preserve the memory value.

## RMBench policy-server handoff

For a trained TTT checkpoint, its saved config already enables the correct TTT
architecture, so do not override its dimensions at serving time:

```bash
export GR00T_PPU_DEVICE=0  # replace with an authorized physical device id
source scripts/ppu/activate_gr00t.sh

python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/rmbench-ttt-l32/checkpoint-N \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555
```

The RMBench client should follow this lifecycle:

```python
client.reset()                 # exactly once at episode start
while not done:
    action, info = client.get_action(observation)
```

`info["ttt"]` contains per-layer diagnostics. Calling `reset()` mid-episode is
also the intended implementation of the mid-reset memory ablation.

When testing the mechanism on an unmodified vanilla checkpoint, add
`--use-ttt` and the desired architecture flags. Such a dynamically added branch
is randomly initialized and is useful only for wiring/identity tests (for
example gate zero), not for RMBench performance claims.

## Minimum evidence before scaling to all RMBench tasks

Use the same trained backbone and seeds for vanilla, short-history, and TTT.
For TTT, run normal memory, mid-episode reset, and key-observation corruption.
Report both success rate and memory diagnostics. A performance increase alone
does not establish that the policy used history.

## Offline memory probe before RMBench rollout

RMBench simulator success rate remains the final metric, but compact checkpoints can first be
screened on held-out expert demonstrations with a paired causal probe. The probe implementation
used for historical experiments is not distributed as a portable command in this repository.

The probe holds the target frame and flow randomness fixed while comparing ordered history,
pre-target memory reset, and shuffled history. Positive `reset_minus_ordered_loss` or
`shuffled_minus_ordered_loss` is evidence that history helps the offline expert-action loss.
Fast-state distances separately reveal whether memory is being written even when the action loss
is unchanged. These are proxy diagnostics and must not be reported as simulator success rates.

Use `--holdout-episodes-per-task N` during training to exclude the deterministic final N episodes
of every task group from optimizer sampling. The offline probe uses the same tail convention.
