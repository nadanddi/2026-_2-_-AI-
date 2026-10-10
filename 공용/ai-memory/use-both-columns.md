---
name: use-both-columns
description: A submission scores temperature and EC separately — never leave one column as the unchanged baseline when proposing a test submission
metadata:
  node_type: memory
  type: feedback
  originSessionId: 96fe1d15-2691-4cbd-a117-4a3e2d96bdf9
  modified: 2026-10-10T03:30:25.122Z
---

The user (2026-10-10) objected when I suggested pairing the EC hypothesis-test submission (submission_15) with the unchanged 9th-round temperature: "9회차 온도로 내면 온도 실험해볼 기회를 1번 잃는거잖아? 제발 생각 좀 하고 말해".

**Why:** Each leaderboard submission reports the temperature RMSE and the EC RMSE separately, and submissions are scarce (deadline 2026-10-16). A column left at the baseline wastes half of a submission's information.

**How to apply:** Whenever a submission is being planned for one target, also propose the most informative candidate for the other target. Give its pre-written expected outcome and how each result would be interpreted. Choose by information value and pass-2 evidence (the test set is all pass 2), not just by the local verdict. The user still decides whether to submit ([[no-submission-prep-until-asked]], [[rule5-interpretation]]).
