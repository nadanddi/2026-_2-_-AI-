---
name: aihub-derived-ok-with-provenance
description: User ruling 2026-10-12 — AI-Hub-derived data (e.g., the 152k colour-fingerprint DB for source lookup) may go into the image submission; README just needs where/how the AI-Hub originals were obtained
metadata:
  type: project
---

User decision (2026-10-12): for the image mission, AI-Hub originals only need their provenance stated (where and how they were obtained) — "ai허브 원본은 어디서 어떻게 가져왔는지만 적어도 돼". So the L2 source-lookup fingerprint DB (AI-Hub 지능형 수직농장 통합 데이터(딸기), 152,043 photos × 12×16 RGB ≈ 88MB) may be shipped in model/.

**Why:** removes the redistribution blocker on the L2 lookup rule (B2 + 5판 + L2 scored 21/21/20 on public 21).

**How to apply:** in the README, state dataset name/number (AI-Hub 596), how it was downloaded, which archives, and how fingerprints were derived; still don't ship the photos themselves. Related: [[lab-claude-image-mission]], [[image-approach-official-types]].

Update 2026-10-12 (later): user confirmed "ai허브 데이터 써도 된다고 했다고" — AI-Hub data use is permitted for the competition; don't raise data-rights/redistribution concerns about it again.
