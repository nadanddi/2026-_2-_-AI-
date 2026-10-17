---
name: claude-advises-codex-runs
description: "Since 2026-10-16 (image mission), Claude only gives opinions/requests; Codex runs experiments and records results, to save tokens"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9650adb3-8c72-4c03-b469-1022ace588e8
  modified: 2026-10-15T17:17:30.310Z
---

2026-10-16: the user said "너는 의견만 내고 실험은 코덱스한테 시키자. 토큰 아껴야 해" (나단9 got LB 0.97). Recording results (성능_정리, 제출기록) also goes to Codex.

**Why:** Claude token budget is tight near the final deadline (10-16 18:00).

**How to apply:** don't launch server runs, subagent critics, or long scripts myself; write a short paste-ready request for 연구실 코덱스 instead and keep replies brief. Related: [[harsh-critic-every-result]] (critique cost now on Codex side), [[image-mission-handoff-1015]].
