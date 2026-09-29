---
name: division-temp-claude-ec-codex
description: From 2026-09-28 Claude works ONLY on temperature (sub_temp); Codex owns EC (sub_ec) improvement
metadata:
  node_type: memory
  type: project
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-28T00:54:40.113Z
---

User decision (2026-09-28): "넌 온도 개선만 집중해서 분석하고 코덱스는 EC 개선에만 집중하게 만들고 싶어".

**Why:** Claude and Codex were duplicating EC work (both reached the same EC candidate 0.8·round-3 + 0.2·TabPFN) and competing for the same CPU; Codex also lacked Claude's temperature findings.

**How to apply:**
- Start no new EC experiments; route EC ideas/findings to Codex via a request file in research/ (Codex_요청_*.md) and the shared catalog research/데이터_단서_카탈로그.md.
- Temperature is Claude's: current candidate G_C2 (catalog 6.70). Avoid heavy CPU jobs at the same time as Codex's runs when possible (GPU is fine).
- Codex works in its own worktree (C:/Users/aozks/.codex/worktrees/...); read its reports there, never edit them.
- Related: [[explore-autonomously]], [[skill-not-luck]], [[no-submission-prep-until-asked]].
