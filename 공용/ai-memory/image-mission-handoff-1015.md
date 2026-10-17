---
name: image-mission-handoff-1015
description: Image mission state at 2026-10-15 handoff — 나단6 LB .95 valid, 나단8 package ready (hypothesis-test submit, revert rule), R4 sealed unopened; read IM_인계_2026-10-15.md first
metadata:
  type: project
---

As of 2026-10-15 17:55 (lab Claude, user switching accounts): read `연구실/클로드/image/IM_인계_2026-10-15.md` first.
- 나단6 = LB 0.95 (current valid). 팜모니_나단8.zip built on user request (H2 experts + JPEG-format gate), awaiting user Colab T4 timing and upload; it is a hypothesis-test submission (pre-registered criterion 7 failed).
- Pre-fixed revert rule: if 나단8 LB ≤ 0.95 → resubmit 나단7 (=나단6 output) next round; never tune 나단8 on LB.
- User says image scoring deadline is 22:00 daily (doc says 18:00 — user's word wins); final 10-16 18:00.
- Sealed R4 (700, server data/local/im_set_r4) unopened — reserve for the next candidate's confirmation. Lab Codex is producing GPT whole-image re-renders → candidate I'.

**Why:** session ran out of quota; continuity across accounts.
**How to apply:** start from the handoff doc; don't redo rejected F2/G/I2/I; keep sealed sets single-use. Related: [[lab-claude-image-mission]], [[lab-gpu-server-rules]].
