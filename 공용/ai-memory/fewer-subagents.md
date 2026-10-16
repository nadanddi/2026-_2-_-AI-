---
name: fewer-subagents
description: User asked (2026-10-12) to minimise subagent use — do the work myself; critic subagent only at the final stage of a real experiment
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0236f83d-274b-48b3-a2d4-9fbedfc12db9
  modified: 2026-10-12T00:27:39.826Z
---

User said on 2026-10-12 (집): "서브에이전트 활용은 왠만하면 줄여보자". This came after a session that delegated many parallel research and critic agents (and three of them died at a usage limit).

**Why:** subagents use up the usage limit quickly, and the work often stalls when the limit is hit.

**How to apply:**
- Write and run the analysis scripts myself. Do not delegate research, forensics or role-play expert panels.
- The [[harsh-critic-every-result]] rule still holds, but in a reduced form:
  - Plan and midpoint checks: do them as my own written self-critique, using the same checklist (leakage, rules, selection bias, small n / multiple testing, bugs, overclaiming).
  - Final stage: spawn one critic subagent only for experiments whose result could change a candidate or an adoption decision.
  - Purely descriptive checks get no critic subagent.
- Use a subagent only if the user asks for one, or if the task truly cannot be done inline. If so, say why.
