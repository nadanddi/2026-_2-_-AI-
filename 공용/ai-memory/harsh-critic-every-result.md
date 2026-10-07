---
name: harsh-critic-every-result
description: "Every experiment gets an independent harsh critic at plan, midpoint and final stages; always report the critique and improvements to the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: f4daec3e-80e9-491e-b23f-b231dc62decb
  modified: 2026-10-06T16:53:13.385Z
---

Every experiment I run must be reviewed by an independent harsh critic (separate subagent reading the plan/code/logs/claims) at **three stages: 계획 단계, 중간 점검 단계, 최종 단계**. After every critique I must tell the user the critique content and the improvement points (format: 결과/계획 → 비평가 평가 → 문제 확인 → 개선책), and apply confirmed fixes (as new file versions) before moving on. Said by the user 2026-10-06/07 (연구실); same rule the user gave 집 코덱스 (`집/코덱스/analysis/critic_workflow_20261006_v1.md`).

**Why:** earlier results were overstated or had hidden flaws (6.349 → 6.350 retraction; LINK3 gain came from future-hour inputs that violated the rules; ND2 v1 had 3 fatal errors the critic caught).

**How to apply:** plan stage — write the plan (hypotheses, data, rules check vs 중간안내 2·3절, fixed criteria) to a file and have the critic attack it before running. Midpoint — after the first partial results/first validator, critic checks for bugs/leakage before the full run. Final — critic reviews code+logs+claims. Ask it to hunt for leakage, rule violations, selection bias, small-n/multiple-testing, code bugs, overclaiming. Related: [[skill-not-luck]], [[analysis-before-answers]], [[no-exaggeration]].
