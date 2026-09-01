# Project status

Updated: 2026-09-01

## Summary

MemGR00T-RMBench is a research preview. The fast-memory mechanism, ordered
sequence path, policy reset/noise protocol, and RMBench absolute-action boundary
are implemented. The focused source-level test suite passes in the maintained
PPU environment after fixing fail-closed sequence collation.

The current public evidence does **not** establish that RoboTTT improves RMBench
task success. The only trained RoboTTT checkpoint currently available is a
historical mechanism pilot derived from a Vanilla-3k parent and older
command-state data.

## Evidence levels

### Verified

- Fast Q/K/V projections, inner K-to-V update, learned gate, and state carry.
- One memory commit per policy query; flow substeps do not multiply updates.
- Ordered sequence training and TBPTT detach without value reset.
- Sequence collation transposes batch-major windows into timestep-major batches.
- Mixed sequence/single-step batches and unequal sequence lengths fail closed.
- Memory reset is independent from policy-noise reseeding.
- Retry of the same `(episode_id, request_id)` is byte-idempotent.
- RMBench v4 returns processor-decoded absolute arm and gripper actions.
- Formal RMBench action offsets are `[0..15]` (H16).

### Historical mechanism evidence

- The pilot checkpoint uses TTT layers 28--31, `ttt_dim=64`,
  `fast_hidden_dim=128`, sequence length 512, and TBPTT segment length 16.
- Offline probes show that fast state is written and repeated runs are
  deterministic under the tested protocol.

### Not yet demonstrated

- A frozen, reproducible actual-qpos + physical-gripper training dataset.
- A selected closed-loop-capable actual-state Vanilla parent.
- A RoboTTT fork from that exact parent with identical slow weights.
- Statistically powered task-success improvement over Vanilla.
- Parity with the private/original RoboTTT architecture and training recipe.

## Release policy

Artifacts are labeled according to evidence:

- `pilot` or `mechanism` — inspectable historical artifact, not a task claim;
- `candidate` — passes identity, conversion, replay, and open-loop gates;
- `validated` — additionally passes paired closed-loop evaluation.

No current checkpoint is labeled `candidate` or `validated`.
