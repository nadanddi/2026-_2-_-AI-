---
name: division-temp-claude-ec-codex
description: Since 2026-10-11 집 클로드 owns BOTH temperature and EC (user: Codex EC too weak); earlier temp-only/EC-only splits superseded
metadata:
  node_type: memory
  type: project
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-28T00:54:40.113Z
---

User decision (2026-09-28): "넌 온도 개선만 집중해서 분석하고 코덱스는 EC 개선에만 집중하게 만들고 싶어".

**Why:** Claude and Codex were duplicating EC work (both reached the same EC candidate 0.8·round-3 + 0.2·TabPFN) and competing for the same CPU; Codex also lacked Claude's temperature findings.

**How to apply:**
- Start no new EC experiments; route EC ideas/findings to Codex via a request file in my own folder (<장소>/클로드/…/Codex_요청_*.md; older ones are in 집/클로드/research/), the "다른 곳에 알릴 것" section of my 작업일지, and the shared catalog 공용/데이터_단서_카탈로그.md.
- Applies to every place: 집 and 연구실 Claude both do temperature; 집 and 연구실 Codex both do EC.
- Temperature is Claude's: current candidate G_C2 (catalog 6.70). Avoid heavy CPU jobs at the same time as Codex's runs when possible (GPU is fine).
- Codex's work is in 집/코덱스/ and 연구실/코덱스/ on branch main (its old worktree / codex-ec branch were merged 2026-09-29); read its reports there, never edit them.
- Related: [[explore-autonomously]], [[skill-not-luck]], [[no-submission-prep-until-asked]].

**Override 2026-10-02 (current):** the user said "클로드 너는 코덱스랑 같이 EC만 쭉 파보자 일단" — Claude now works on EC together with Codex, temperature paused (a teammate is building the temperature model). Split EC work with Codex through request files in my folder so the two do not duplicate (e.g. Codex = TabPFN/v2 pipeline and its artifacts, Claude = post-processing/stacking/validator evidence on Codex's saved OOF). Revert to the temperature-only rule only when the user says so.

**Override 2026-10-11 (CURRENT, supersedes the above):** the user said "그냥 EC 온도 둘 다 집 클로드가 해버리자. 코덱스 성능이 너무 별로야." The user judges Codex's EC results weak. 집 클로드 now owns BOTH temperature and EC (model work, experiments, submission columns). Codex output stays readable as reference (read-only, never edit its folders). There is no need to route EC ideas to Codex via request files any more. The root CLAUDE.md / AGENTS.md / 00_먼저_읽기.md still state the old split; the user said NOT to edit them ("파일 굳이 고치지 마 귀찮다", 2026-10-11). Follow this memory and the HANDOFF note instead.
