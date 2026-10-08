---
name: prior-competition-data-restricted
description: "User's previous-competition strawberry data is BANNED — never load, analyse, model with, or cite it"
metadata:
  type: project
  modified: 2026-10-08
---

The user holds data from a previous competition (`train_X.csv.xlsx`, `train_y.csv`). **Do not use it in any way**: do not open, load, analyse, compare against, model with, derive features/weights/flags from, or cite findings from it. This covers both training and "diagnosis only" uses.

**Why:** fairness, rule 8.1 (data provenance); organizer said teams using prior-competition data get penalised.
**How to apply:** never suggest using it or asking permission again. Do not write any history or narrative about this data (when it was allowed/banned, what was analysed or deleted) in worklogs, HANDOFF, catalog, or memory — only the ban itself. If the files or analyses of them appear inside the repo, keep them out of git (`sync_end.ps1` runs `git add -A`; the data is no-distribution) and tell the user. Related: [[rule5-interpretation]], [[no-exaggeration]].
