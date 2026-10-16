---
name: structure-matched-validator
description: 2026-10-14 user approved building an eval-structure-matched validator (pseudo-eval blocks) plus situation-stratified judging
metadata:
  node_type: memory
  type: project
  originSessionId: 5c891cda-121c-4938-81fb-c5770ce2797d
  modified: 2026-10-13T17:39:20.223Z
---

On 2026-10-14 the user said "둘 다 묶어서 만들어봐" after I proposed (1) an eval-structure-matched validator and (2) situation-stratified judging, and flagged that mimicking the eval block structure touches the organiser ban on "tuning to the evaluation set's statistics/composition".

**Decision:** building a validator that mimics the eval record-block structure is accepted, as EV and mirror CV already did. The model itself must still never use eval statistics, and validation results must never be weighted by eval situation counts.

**Why:** existing validators disagreed. DIAG10 and EV are 13–14 % pass-2, while eval is 100 % pass-2. P2LOO and EL1 resemble eval but hold only 6–8 high-EC records. Pure-window and deleted-boundary situations were barely represented (VC1, catalog 6.825).

**How to apply:**
- Pseudo-eval draws hide same-shape chunks, pass-2 first, with deleted boundaries mirroring eval and some same-date other-ID records hidden.
- Recompute structure-dependent parts (linker, calendar, SG2 references) per draw.
- Check the situation shares against eval before use.
- Judge per situation: the method's target situation must improve, and the other situations must show no harm.

Related: [[skill-not-luck]], [[rule-safety-checklist]].
