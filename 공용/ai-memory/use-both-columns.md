---
name: use-both-columns
description: A submission scores temperature and EC separately — never leave one column as the unchanged baseline when proposing a test submission
metadata:
  node_type: memory
  type: feedback
  originSessionId: 96fe1d15-2691-4cbd-a117-4a3e2d96bdf9
  modified: 2026-10-10T12:24:24.967Z
---

The user (2026-10-10) objected when I suggested pairing the EC hypothesis-test submission (submission_15) with the unchanged 9th-round temperature: "9회차 온도로 내면 온도 실험해볼 기회를 1번 잃는거잖아? 제발 생각 좀 하고 말해".

**Why:** Each leaderboard submission reports the temperature RMSE and the EC RMSE separately, and submissions are scarce (deadline 2026-10-16). A column left at the baseline wastes half of a submission's information.

Repeated 2026-10-10: I proposed replacing a column whose outcome is already arithmetically fixed with the 9th-round baseline. The user objected again: "왜 자꾸 테스트할 수 있는 기회를 이미 테스트 끝난 애들로 채우려는 거야". A baseline column (known LB score) is as wasteful as a predetermined one. Every column must carry an UNTESTED hypothesis whose outcome is not already implied by past scores. Check this with the score identity before proposing.

**How to apply:** Whenever a submission is being planned for one target, also propose the most informative candidate for the other target. Give its pre-written expected outcome and how each result would be interpreted. Choose by information value and pass-2 evidence (the test set is all pass 2), not just by the local verdict. The user still decides whether to submit ([[no-submission-prep-until-asked]], [[rule5-interpretation]]).

Third time 2026-10-11: after round 10 (EC 0.3538) I proposed reverting the next EC column to the 9th-round EC "because the last valid submission is the final answer". User: "다음 제출 EC를 왜 9회차로 돌리냐고 제발 생각 좀 해봐". The final-answer rule only constrains the LAST submission before the deadline. Every intermediate submission tests new hypotheses in BOTH columns. Reserve one final slot just before the deadline (2026-10-16 18:00) for the best-known combination (best temp column + best EC column, which may come from different rounds). Never use "the final answer is currently bad" as a reason to fill a column with a known baseline.
