---
name: no-submission-prep-until-asked
description: "Do not propose/declare \"확정된 N회차 구성\" or build submission files until the user explicitly says to make the next submission"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-26T17:17:10.384Z
---

The user said (2026-09-27): "내가 6번째 제출을 하자고 말하기 전까진 '확정된 08 구성' 이런 거 만들지 마."

**Why:** The user decides when a submission happens; framing analysis results as a finalized submission config (or offering to build the CSV/ZIP/설명자료 triple) pushes toward spending a scarce slot prematurely.

**How to apply:** Report validation results as findings/candidates only ("후보", "검증 결과"). Do not write sections like "확정된 08 구성", do not ask "08을 만들까요?", and do not create submission generators/packages until the user explicitly asks for the next submission. When asked, then follow [[deliverable-triple]] and [[new-files-not-edits]].
