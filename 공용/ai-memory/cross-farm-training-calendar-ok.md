---
name: cross-farm-training-calendar-ok
description: 2026-10-11 user ruling — the OTHER greenhouse's TRAINING records (train_X inputs, train_y labels) may be used to build the date calendar / linker for evaluation rows; other-farm EVALUATION inputs stay banned
metadata:
  type: project
---

User decision 2026-10-11 ("허용해"): when identifying an evaluation record's date / same-house previous date, the linker may use the other greenhouse's (F13 <-> F47) training records — their outdoor inputs for date matching and calendar order, and their labels. Evaluation inputs of the other greenhouse remain forbidden, as do later-numbered evaluation inputs of the same greenhouse.

**Why:** F13 and F47 share the same site weather. A date that has no pass-1 record in F47 can often be placed only via F13's training records (e.g., F47_191's true previous date F47_057, catalog 6.567). The official guidance (catalog 6.309) allows models that store and reference all public train_X / train_y.

**How to apply:** Build cross-farm calendars only from train_X/train_y. Keep the hour-causal rule for the current record (0..h inputs). Run isolation checks so that perturbing the other farm's TEST inputs never changes predictions. Related: [[rule5-interpretation]], [[training-cleaning-noncausal-ok]].
