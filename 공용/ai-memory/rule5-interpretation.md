---
name: rule5-interpretation
description: "User's reading of competition rule section 5 — only direct access to hidden labels (hacking/leaks) is banned; analysing leaderboard scores is allowed"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-10-05T14:56:03.405Z
---

Problem statement section 5 says "비공개 평가 정답에 접근하거나 이를 학습·특징 구성에 사용하는 행위는 금지". The user (team member deciding for the team) interprets this as a ban on obtaining the hidden label data itself (e.g. hacking the server, leaked files). Using the official leaderboard score feedback analytically — e.g. solving the four submissions' scores for segment-level bias, reading score changes as validation — is acceptable to them. Decided 2026-09-26.

**Why:** I had been refusing to use score-based inference as possible "evaluation-label access"; the user corrected this.

Also decided 2026-10-01: public **train_y labels may be used as features for evaluation rows** (e.g. the state/offset of labelled train days adjacent to a test block). User said "그거 써도 돼" when I flagged this as ambiguous (설명회 12쪽 only says train_y may be used for model training). Default to earlier-day labels; flag later-day (future-direction) label use separately before relying on it.

Reconfirmed 2026-10-02: **never use leaderboard scores to choose blend weights or model directions for a submission**, even when the score identity shows a big gain (EC .2055→.153 was computable). The user said the final presentation must explain why the model was built and what data it used, and "we tuned on the leaderboard" cannot be presented honestly. Leaderboard analysis stays diagnostic only (e.g. which validator the eval follows).

**Official clarification 2026-10-05 (organizer 중간 안내 PDF, catalog 6.309):** models trained on / storing and referencing ALL public train_X·train_y are allowed, and the time limit does NOT restrict training-data reference to before the eval row → later-day train_y labels may be referenced. Banned: any input later than the row at ANY step (features, preprocessing, search, linking, correction, post-processing); fitting/updating anything (incl. preprocessing such as a calendar) on test_X or its predictions; adjusting to test-set-wide statistics. Final answer invalid → latest of last 3 valid submissions that passes verification is used (not best score) → keep code/model/settings for every submission.

**Organizer 추가 안내 (user relayed 2026-10-05, catalog 6.344):** comparing submissions by public score is allowed; computing blend weights or correction coefficients from earlier evaluation-prediction files + public scores is NOT allowed (final answer invalid; fix before deadline). 25 submissions are for trying ideas, not for learning hidden labels. → Leaderboard-derived numbers must never set a weight/coefficient/threshold; avoid even letting LB-inferred test composition steer method choice.

**How to apply:** leaderboard-score analysis is allowed for aggregate/segment-level diagnosis, not for building or selecting the submitted model. Still avoid back-solving individual test labels and hard-coding them into answers, since section 8.2 reproduction review reads the code. Related: [[analysis-before-answers]]
