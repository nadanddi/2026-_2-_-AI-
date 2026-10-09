---
name: plain-explanations
description: "Explain findings in plain words first (what it means), numbers second; user pushed back on jargon-heavy reports"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4b0e6aba-2304-4718-b0b9-75092c2b57d6
  modified: 2026-10-08T19:45:36.105Z
---

When reporting a finding, lead with a plain-language explanation of what it means and why it matters, then give the numbers.

**Why:** 2026-10-09 the user replied "뭔소리야. 뭐가 의미 있다는 거야 쉽게 말해봐" to a dense report about a validation leak, and later caught that my claim ("검증이 낙관적") contradicted the actual numbers (validation worse than leaderboard). Dense jargon hid the logic error from both of us.

**How to apply:** Use everyday analogies (e.g. "연습 시험에 실제 시험에 없는 힌트가 있었다"), say explicitly whether a claim is confirmed or only possible, and check the claim's direction against known numbers (e.g. LB vs validation) before stating it. Related: [[no-exaggeration]], [[analysis-before-answers]].
