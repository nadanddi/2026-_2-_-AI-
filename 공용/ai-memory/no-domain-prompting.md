---
name: no-domain-prompting
description: Don't mention or ask for domain knowledge (greenhouse operations, sensor setup, expert questions) until the user hands over materials
metadata:
  type: feedback
---

Do not bring up domain knowledge — no expert-question lists, no "waiting for domain answers", no "ask your team" — until the user explicitly provides domain materials.

**Why:** 2026-10-01 the user said "도메인 지식에 대해서는 내가 자료 넘겨줄 때까지 언급하지 마" after I repeatedly ended reports with domain questions. They are gathering it themselves and the reminders are noise.

**How to apply:** end reports with data-driven next steps only. When the user shares domain material, use it immediately (build a fingerprint table from it per the data-generation-forensics skill). Related: [[no-presentation-prep]], [[explore-autonomously]].
