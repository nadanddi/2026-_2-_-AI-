---
name: record-number-is-official-day
description: "row_id number = official 'greenhouse relative day' (problem statement); number-order history features are field-consistent, not a competition artifact"
metadata:
  node_type: memory
  type: project
  originSessionId: b095b317-efbd-4cc0-ba3f-869557279353
  modified: 2026-10-16T05:22:46.484Z
---

Problem statement (정형데이터_문제설명서.pdf §2): "row_id 는 온실 ID, 온실별 상대 일차, 하루 안의 시각… F47_183_00 은 F47 온실의 183 일차 0 시". The record number is the OFFICIAL per-greenhouse relative day index.

**Why:** On 2026-10-16 I wrongly told the user that BASE's number-ordered cross-day history was a 5-point (field applicability) 0-point risk. The user pointed out it is the official relative day. Using the official time axis is field-consistent: the field knows "day N of this greenhouse". The data's inconsistencies (same house switching F13/F47 IDs, number neighbours often not the same house) are a data problem that HURTS the model, not a regularity it exploits.

**How to apply:** When auditing 5-point fitness, treat features ordered by the relative day (lags, rolling, previous-day aggregates) as legitimate. Flag only things with no field meaning: number-difference link rules (v7/v10/v15R 'number −2'), the 179 boundary as a phase marker, future neighbours' labels, cross-farm test inputs. Present the FL linker as "finding the true same-house yesterday where the data is inconsistent with the official day index". Related: [[field-applicability-rubric]], [[no-exaggeration]].
