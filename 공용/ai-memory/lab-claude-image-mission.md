---
name: lab-claude-image-mission
description: "Since 2026-10-10 연구실 Claude works all-out on the IMAGE mission (real vs generated photo), independent of 집 Codex"
metadata:
  node_type: memory
  type: project
  originSessionId: 72cf93e1-8c65-4dfd-8614-7baba25d7126
  modified: 2026-10-10T11:11:04.709Z
---

User decision (2026-10-10, 연구실): "너는 이제 이미지 미션 해결에 총력을 다할거야". Clarified: applies to **연구실 클로드 only** (집 클로드 keeps temperature); relation to 집 코덱스 (already doing image since 10-09) = **independent parallel** — read Codex's image results as reference, build my own baseline/validation quickly, results cross-check each other.

**Why:** 집 코덱스 spent 10-09~10-10 on plans/audits (PLAN v1~v8) with no scored baseline yet; user wants a second independent push.

**How to apply:**
- My files go in `연구실/클로드/` (e.g. `image/`), heavy outputs in `연구실/클로드/local/`.
- Codex image work: `집/코덱스/analysis/image_*` (read only). Don't duplicate its 350GB ingestion audit; reuse facts (e.g. 배포 Validation shares groups with Training → not an independent validator).
- Same adoption discipline applies ([[skill-not-luck]]), no submission files until user says ([[no-submission-prep-until-asked]]).
- Overrides [[division-temp-claude-ec-codex]] for 연구실 Claude only.
