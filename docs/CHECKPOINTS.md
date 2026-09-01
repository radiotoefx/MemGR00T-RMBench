# Checkpoints

## RoboTTT pilot checkpoint-1000

The first project release contains the historical mechanism pilot from:

```text
gr00t-rmbench-robottt-swap-vanilla3k-last4-joint-w16-single-1k/checkpoint-1000
```

Configuration summary:

```text
use_ttt=true
ttt_layer_indices=[28,29,30,31]
ttt_dim=64
ttt_hidden_dim=128
ttt_sequence_length=512
ttt_tbptt_segment_length=16
ttt_action_prefix_chunk_size=16
ttt_decision_loss_weight=16
```

The archive includes the compact RoboTTT trainable weights, model config,
processor config, statistics, embodiment identity, trainer state, and training
metadata. Checksums are published beside the archive in the GitHub prerelease.

### Critical limitation

This is **not a standalone checkpoint**. Its `trainable_model.json` declares:

```text
base_model_path = GR00T-N1.7-3B
parent_checkpoint = gr00t-rmbench-swap-projector26-b8-cont3k/checkpoint-3000
```

The 6.1 GiB historical Vanilla-3k parent is not included in the pilot release.
Loading only the small RoboTTT delta on the public base does not reconstruct the
evaluated slow weights and must not be reported as the original pilot.

### Intended use

- inspect tensor names, architecture, processor, and training metadata;
- reproduce compact-checkpoint loading when the declared parent is available;
- test memory/reset/diagnostic wiring;
- serve as provenance for future migration tools.

It must not be used as evidence of closed-loop task improvement or as a formal
actual-state RMBench model.

## Vanilla60k

The currently deployed Vanilla60k inference weight file is approximately
3.24 GiB and the complete training checkpoint is approximately 30 GiB. It is
not bundled into this source repository or the pilot release. It is also not a
RoboTTT checkpoint.

Large standalone model distribution will use a model-oriented host or split
release assets with immutable SHA256 manifests; weights will not be committed
to the Git repository.
