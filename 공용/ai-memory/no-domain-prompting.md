---
name: no-domain-prompting
description: "Don't mention or ask for domain knowledge (greenhouse operations, sensor setup, expert questions) until the user hands over materials"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8d280fa9-d7f5-4e80-be3a-327f3fe6102c
  modified: 2026-10-03T10:04:43.899Z
---

Do not bring up domain knowledge — no expert-question lists, no "waiting for domain answers", no "ask your team" — until the user explicitly provides domain materials.

**Why:** 2026-10-01 the user said "도메인 지식에 대해서는 내가 자료 넘겨줄 때까지 언급하지 마" after I repeatedly ended reports with domain questions. They are gathering it themselves and the reminders are noise.

**Received so far:** 2026-10-01 the user relayed a domain person's view (grower side): EC — they don't know; temperature — substrate may be weakly tied to inside air (underground is cold even when the surface is warm), most tied to irrigation water temperature; an outdoor tank follows outdoor temp/sunlight, an indoor tank follows inside temp. The user does not want to export gallery memos one by one. Tested as catalog 6b.39 (calendar-ordered tank memory) → rejected. Don't re-ask for this; build on it.
2026-10-03 the user handed two more files (copied to 집/클로드/research/domain_2026-10-03/): a variable-effect comment list (curtains/fog/humidity → substrate water↑ EC↓, inside temp↑ → EC↑, CO2 → water↓) and a student talk (drainage EC ↑ in reproductive stage, ↓ in vegetative; target 1.2–1.5). Episode periodicity tested (PH0) → no consistent period. The 09-26 Downloads report "딸기 수경재배 배지 EC 예측 현장 지식 조사" is a web-research report, not the researcher's material.

**How to apply:** end reports with data-driven next steps only. When the user shares domain material, use it immediately (build a fingerprint table from it per the data-generation-forensics skill). Related: [[no-presentation-prep]], [[explore-autonomously]].
