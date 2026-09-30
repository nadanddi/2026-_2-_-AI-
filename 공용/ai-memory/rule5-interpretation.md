---
name: rule5-interpretation
description: "User's reading of competition rule section 5 — only direct access to hidden labels (hacking/leaks) is banned; analysing leaderboard scores is allowed"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-30T18:52:02.047Z
---

Problem statement section 5 says "비공개 평가 정답에 접근하거나 이를 학습·특징 구성에 사용하는 행위는 금지". The user (team member deciding for the team) interprets this as a ban on obtaining the hidden label data itself (e.g. hacking the server, leaked files). Using the official leaderboard score feedback analytically — e.g. solving the four submissions' scores for segment-level bias, reading score changes as validation — is acceptable to them. Decided 2026-09-26.

**Why:** I had been refusing to use score-based inference as possible "evaluation-label access"; the user corrected this.

Also decided 2026-10-01: public **train_y labels may be used as features for evaluation rows** (e.g. the state/offset of labelled train days adjacent to a test block). User said "그거 써도 돼" when I flagged this as ambiguous (설명회 12쪽 only says train_y may be used for model training). Default to earlier-day labels; flag later-day (future-direction) label use separately before relying on it.

**How to apply:** leaderboard-score analysis is allowed for aggregate/segment-level decisions. Still avoid back-solving individual test labels and hard-coding them into answers, since section 8.2 reproduction review reads the code. Related: [[analysis-before-answers]]
