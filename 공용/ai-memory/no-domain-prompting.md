---
name: no-domain-prompting
description: "Don't mention or ask for domain knowledge (greenhouse operations, sensor setup, expert questions) until the user hands over materials"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8d280fa9-d7f5-4e80-be3a-327f3fe6102c
  modified: 2026-10-01T05:23:37.496Z
---

Do not bring up domain knowledge — no expert-question lists, no "waiting for domain answers", no "ask your team" — until the user explicitly provides domain materials.

**Why:** 2026-10-01 the user said "도메인 지식에 대해서는 내가 자료 넘겨줄 때까지 언급하지 마" after I repeatedly ended reports with domain questions. They are gathering it themselves and the reminders are noise.

**Received so far:** 2026-10-01 the user relayed a domain person's view (grower side): EC — they don't know; temperature — substrate may be weakly tied to inside air (underground is cold even when the surface is warm), most tied to irrigation water temperature; an outdoor tank follows outdoor temp/sunlight, an indoor tank follows inside temp. The user does not want to export gallery memos one by one. Tested as catalog 6b.39 (calendar-ordered tank memory) → rejected. Don't re-ask for this; build on it.

**How to apply:** end reports with data-driven next steps only. When the user shares domain material, use it immediately (build a fingerprint table from it per the data-generation-forensics skill). Related: [[no-presentation-prep]], [[explore-autonomously]].
