---
name: check-catalog-before-new
description: "Before calling a hypothesis \"untested\" or proposing it as new, grep 공용/데이터_단서_카탈로그.md (and Codex/lab worklogs) for prior tests"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3bdf8982-4d48-496d-86ba-7a6ea55662d2
  modified: 2026-09-30T18:19:22.171Z
---

Before telling the user a direction "has never been tested" or proposing it as a fresh hypothesis, search `공용/데이터_단서_카탈로그.md` (and the other three folders' 작업일지) for it first.

**Why:** On 2026-10-01 I told the user source(동)-level day offsets were never directly verified and proposed it as the next experiment; the catalog already had 6b.3, 6b.20 (lab Claude) and Codex H13/H14/6.81 rejecting it. Four places × two AIs produce many experiments; repeating one wastes the user's limited time before the 10-16 deadline and erodes trust.

**How to apply:** grep the catalog with 2–3 Korean keywords of the idea (e.g. 출처, 이웃, 형제, 사전학습) before stating novelty; if prior work exists, cite its number and say what would be genuinely different. Related: [[analysis-before-answers]], [[skill-not-luck]].
