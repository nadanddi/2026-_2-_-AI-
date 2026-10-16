---
name: no-premature-best
description: "Don't call a configuration \"최선/상한\" — say \"현재 검증된 최고 구성\" and list the remaining improvement room"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b095b317-efbd-4cc0-ba3f-869557279353
  modified: 2026-10-15T21:03:00.883Z
---

User (2026-10-16): "자꾸 최선이 바뀌는 것 같은데 개선 여지가 있는걸 왜 자꾸 최선이라고 하는거야?" — after I repeatedly called the current temperature config "최선" (27+GRU30, then 27+chain) while new improvements kept appearing.

**Why:** "최선" reads as "cannot be improved", which overstates the evidence and contradicts later findings; related to [[no-exaggeration]].

**How to apply:** Describe the current pick as "현재 검증된 최고 구성" (best that passed validation so far). Whenever giving a status, add which directions still have room (with their evidence) and which are closed by evidence (and what new information would reopen them). Never say "상한/더 짜낼 게 없다" unless an oracle/upper-bound analysis actually shows it.
