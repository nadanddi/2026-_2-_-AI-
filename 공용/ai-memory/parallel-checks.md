---
name: parallel-checks
description: Run the submission rule-check reruns (base + future cuts + farm isolation) in parallel, not one after another
metadata:
  type: feedback
---

The user said (2026-10-15): "앞으로 병렬로 돌려" — after checks_v28 took over an hour running six full pipeline reruns one after another.

**Why:** Each rerun takes about 12 minutes, most of it the deterministic CPU GRU. Run in sequence, six reruns block the ZIP and the explanation doc for over an hour.

**How to apply:** In checks_vNN.py, launch the base run and every perturbed run at the same time, then compare.
- Use a thread pool. Each run already works in its own temp dir and starts its own subprocesses.
- Keep each step in a separate process and keep the bit-identical comparisons unchanged.
- Check the CPU core count and cap the workers so the runs don't starve each other.
Related: [[deliverable-triple]].
