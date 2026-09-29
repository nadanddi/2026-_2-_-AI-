---
name: skill-not-luck
description: "User wants improvements proven robust (seeds, multiple validators, pre-set criteria), not lucky single-run or single-submission wins"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-27T05:57:47.762Z
---

The user said (2026-09-27): "모델이 운에 의해 맞추는게 아닌 철저한 실력으로 정답을 맞추는 게 중요한 것 같아."

**Why:** Inspector audit showed an EC "−0.9%, 5/5 folds" gain flipped to +0.18% just by changing the random seed, and one leaderboard result (e.g. round 5 −3.0%) is a single noisy observation. The user values real, reproducible skill over lucky scores.

**How to apply:**
- Evaluate every candidate with ≥2–3 random seeds (average + spread) and on several validators (DIAG10 non-overlapping, EXT10/EXT12, EC geometry A); report block-bootstrap CIs.
- Fix decision criteria and blend weights BEFORE looking at results; label post-hoc findings as such.
- Adoption rule (user chose 2026-09-27, replacing the old flat "<1% = 판별 불가"): adopt REGARDLESS of size if the gain has the same direction for every seed × every validator AND the DIAG10 CI excludes 0; when k variants are tested at once, use a Bonferroni-tightened CI (level 1−0.05/k). Label sub-1% passes "작지만 확실한 개선"; when stacking several, re-validate the combination.
- Prefer changes with a physical/agronomic mechanism and consistent direction across subsets (farm × pass, cold days, hours) over changes that win only on one slice.
- Related: [[analysis-before-answers]], [[rule5-interpretation]].
