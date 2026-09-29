---
name: deliverable-triple
description: "Every model build for the agri AI competition must ship three files together — submission CSV, reproduction ZIP, and a Korean explanation document for teammates"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-25T17:27:09.913Z
---

Whenever a new model/submission candidate is made, always produce all three, under the same version number. Since the 2026-09-29 reorganisation they go in 제출/NN회차_YYYY-MM-DD/ (not yet submitted: 제출/미제출/<name>/); before that they were in research/submissions/:
1. `submission_0N.csv` (answer)
2. `팜모니_정형데이터_재현패키지_0N.zip` (reproduction package, rule 8.2)
3. `설명자료_0N_제출K회차.md` (Korean explanation for teammates: what changed, why — the data evidence, model config, CV and real scores, lessons, next step)

Also add a line to 제출/제출기록.md mapping file number ↔ platform submission round (file numbers run one ahead of rounds: submission_03 = round 2, submission_04 = round 3), and a row + section to 제출/성능_정리.md (score, change vs previous round, what it aimed to improve).

**Why:** the user shares model progress with two teammates (6 personal submissions each + 7 shared) and needs to explain each improvement to them.

**How to apply:** don't report a model as done until all three exist and the ZIP is verified from a clean extraction. Related: [[new-files-not-edits]]
