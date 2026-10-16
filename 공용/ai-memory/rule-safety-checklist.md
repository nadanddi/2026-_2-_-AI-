---
name: rule-safety-checklist
description: Pre-submission rule checklist (2026-10-11) — constructs that must NOT appear in any submitted model; check every new model against it
metadata:
  node_type: memory
  type: feedback
  originSessionId: 855aba50-284e-4fd4-81e4-e29dfdb0899a
  modified: 2026-10-12T03:25:29.798Z
---

User (2026-10-11): "하면 안되는 거 잘 기억해두면서 개선에 집중하자. 다음에 제출해볼 모델을 조심하면 되지 않을까?" after pointing at the official bans: (a) learning or updating a model, preprocessing or decision criteria from evaluation inputs or their predictions; (b) adjusting predictions or criteria to statistics, distribution or composition of the whole evaluation set.

**Why:** if the final answer is voided, the first passing submission among the latest 3 is used, or the team gets 0. The clean fallback submission_20 (linker v5 anchors) is held in reserve.

**How to apply:** before building any submission, check each item. A model must not contain any of these:
1. Sequential "claimed" assignment (an earlier evaluation record's decision removing candidates for a later one). UPDATED 2026-10-12 (user showed notice §2 table): a feature COMPUTED from a same-ID earlier evaluation record's inputs with a model trained on public training data (e.g. that record's predicted or anchor-corrected 23 h value) IS allowed — "이전 구간의 입력으로 만든 특징" + "train 전체로 학습한 모델"; it must be recomputable from those inputs + training data only, and never use other-ID evaluation records.
2. Rules designed from the evaluation-set composition, such as the "deleted record (r−1 missing)" branch of W2, or anything keyed to evaluation chunk positions.
3. Normalisation, statistics or weight training that touches evaluation inputs. Training-row features are built in the MASK world.
4. Test values inferred from leaderboard scores used as evidence for a rule (see [[rule5-interpretation]]).
5. Other-farm evaluation inputs, or same-farm later-numbered inputs, used anywhere. Training records of any farm/date are OK (see [[train-data-unrestricted]]). REAFFIRMED by the user on 2026-10-12: "뒤 입력은 어떤 방식으로도 쓰면 안돼". This means no use of later evaluation inputs at all, including only for linking, date or house decisions. Do not propose asking the organizers about it again.
6. Constants chosen by looking at evaluation records. Choose them on training data only.

Every package must also pass: future-cut checks, two-farm isolation checks, and a full-pipeline reproduction. Keep one clean submission among the final 3.

ORGANIZER ANSWER 2026-10-16 (user relayed; full text in 집/클로드/주최측_답변_2026-10-16.md):
- W2 (branching on whether the same house's earlier-day record exists in the eval set / row_id gap) = uses whole-eval composition → BANNED. Submissions with W2: 18, 19, 22, 23, 24. W2-free: 20, 25, 27, 28.
- ALLOWED: values a fixed rule produced for a same-house, earlier-row_id eval record (prediction, link decision) as fixed-formula input for later records — no later inputs, no other-house eval inputs, no model/criterion updates from them. ("prediction chain" is legal.)
- ALLOWED: validators that mimic the eval row_id layout inside training data. Banned: fitting predictions/criteria to eval inputs, predictions or statistics, or tuning to specific eval days.
