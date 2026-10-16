---
name: train-data-unrestricted
description: 2026-10-12 user ruling — training data (train_X/train_y) may be used in any way/direction; test_X restrictions unchanged
metadata:
  type: project
---
User (2026-10-12): "train은 어떻게 써도 상관 없어". Training records' labels and inputs may be used in any direction and role: later-dated training records (same or other greenhouse ID) as reference/anchors, other-farm TRAINING inputs as eval-row feature values, label continuity, structure.

**Why:** organiser notice 10-05 §2 says training-data model learning/reference is not limited to times before the eval row; user decided not to wait for an organiser answer on this point.

**How to apply:** do not block ideas only because they use later or other-farm TRAINING data. Still banned: evaluation inputs later than the row (any step incl. linking), other-farm EVALUATION inputs, eval-composition rules, eval statistics, LB-inferred labels ([[rule-safety-checklist]]). Supersedes the narrower [[cross-farm-training-calendar-ok]].
