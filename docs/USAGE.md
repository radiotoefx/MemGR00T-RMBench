# Using MemGR00T-RMBench

MemGR00T-RMBench keeps the upstream `gr00t` Python package and command-line
interfaces. Existing GR00T checkpoints, processors, embodiment tags, and
Policy API integrations therefore remain usable. The MemGR00T additions are
opt-in through a checkpoint configuration or the RoboTTT options described in
[`robottt_rmbench.md`](robottt_rmbench.md).

## 1. Install

The default environment targets Linux, Python 3.12, CUDA 12.8, and a GPU with
at least 16 GiB of memory for inference. Fine-tuning normally needs 40 GiB or
more. Platform-specific CUDA and edge-device instructions are in the upstream
deployment guide at [`scripts/deployment/README.md`](../scripts/deployment/README.md).

Install Git LFS before cloning so a normal checkout can retrieve the optional
demo data:

```bash
git lfs install
git clone --recurse-submodules https://github.com/radiotoefx/MemGR00T-RMBench.git
cd MemGR00T-RMBench
uv sync --python 3.12
```

For a source-only checkout (for example, CI or a machine without the demo
datasets), skip LFS smudging explicitly:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/radiotoefx/MemGR00T-RMBench.git
cd MemGR00T-RMBench
uv sync --python 3.12
```

Run `git lfs pull` later if you need the demo datasets or LFS-hosted platform
wheels. A source-only checkout can run the lightweight CI checks without them.

The VLM backbone used by the public N1.7 checkpoints is gated on Hugging Face.
Request access to `nvidia/Cosmos-Reason2-2B` and authenticate before loading a
checkpoint:

```bash
uv run hf auth login
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

## 2. Standard GR00T inference

The unchanged GR00T path remains the recommended first smoke test. The DROID
sample works with the public base checkpoint:

```bash
uv run python scripts/download_droid_sample.py --num-episodes 3
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 2 \
  --inference-mode pytorch \
  --execution-horizon 8
```

Other official examples are kept in [`examples/`](../examples/), including
DROID, LIBERO, SimplerEnv, RoboCasa, and SO100. The full Policy API, input
schema, action schema, and embodiment-tag table are documented in
[`getting_started/policy.md`](../getting_started/policy.md).

## 3. Python Policy API

```python
from gr00t.policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.7-3B",
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    device="cuda:0",
    strict=True,
)
action = policy.get_action(observation)
```

Use `policy.reset()` exactly once at the beginning of each episode when using a
MemGR00T checkpoint. Resetting memory and reseeding policy noise are separate
operations; the service protocol exposes both explicitly.

## 4. Server deployment

The standard GR00T server remains available:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555
```

For the RMBench boundary, use the project launcher. It requires explicit device,
seed, checkpoint, action/state semantics, and writes a launch manifest:

```bash
export GR00T_SERVER_DEVICE=0
export GR00T_SERVER_CHECKPOINT=/path/to/checkpoint
export GR00T_SERVER_POLICY_SEED=1234
export GR00T_SERVER_MANIFEST_PATH=/path/to/launch-manifest.json
bash scripts/ppu/run_gr00t_rmbench_policy_server.sh
```

Compact RoboTTT checkpoints already contain their architecture configuration;
do not override their TTT dimensions at serve time. See
[`CHECKPOINTS.md`](CHECKPOINTS.md) for parent-checkpoint requirements.

## 5. Fine-tuning

Prepare data in the GR00T-flavored LeRobot v2 format and follow
[`getting_started/data_preparation.md`](../getting_started/data_preparation.md).
The standard custom-embodiment entry point is:

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /path/to/modality_config.py \
  --output-dir /path/to/output
```

For ordered sequence training and the MemGR00T options, use the reproducible
recipe in [`robottt_rmbench.md`](robottt_rmbench.md) and read
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) before interpreting results.

## 6. Verification

Run the lightweight mechanism and protocol tests after installation:

```bash
uv sync --extra dev
uv run pytest -q \
  tests/gr00t/model/test_robottt.py \
  tests/gr00t/model/test_robottt_sequence.py \
  tests/gr00t/policy/test_policy_service.py \
  tests/gr00t/policy/test_rmbench_adapter.py
```

The repository's larger benchmark and GPU tests retain the upstream markers
and are intentionally not run by the lightweight CI job.

On a PPU environment, select an authorized physical device and optionally point
`GR00T_ENV` at an existing virtual environment before running the same suite:

```bash
export GR00T_PPU_DEVICE=0  # replace with an authorized physical device id
export GR00T_ENV=/path/to/gr00t-venv  # omit when using the repository .venv
source scripts/ppu/activate_gr00t.sh
python -m pytest -q \
  tests/gr00t/model/test_robottt.py \
  tests/gr00t/model/test_robottt_sequence.py \
  tests/gr00t/policy/test_policy_service.py \
  tests/gr00t/policy/test_rmbench_adapter.py
```
