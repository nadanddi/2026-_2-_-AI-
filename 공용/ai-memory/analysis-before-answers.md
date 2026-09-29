---
name: analysis-before-answers
description: "In the agri AI competition, a new finding should first drive deeper data analysis, not an immediate new submission variant"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-26T05:57:43.173Z
---

When a data finding appears, first work out how it lets us analyse and use the data better (what it explains, what new questions it opens, where the remaining error lives) before turning it into a new model/submission version.

**Why:** the user said data analysis is the most important thing. Round 4 (cold hinge) was built straight from a thin finding (5 training rows) and made temperature worse (0.5624 -> 0.5697); the user wants understanding to lead, answers to follow. Only 2 personal submissions remain.

**How to apply:** for each finding, write down what it implies about how the data was generated and which analyses it enables; run those analyses; only propose a submission when a change passes the calibrated validators (temperature: EXTRAP fold holding out all days colder than 10 C; EC: geometry fold set A) and does no harm on the secondary ones. Related: [[deliverable-triple]], [[new-files-not-edits]]
