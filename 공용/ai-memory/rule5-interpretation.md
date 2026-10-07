---
name: rule5-interpretation
description: "User's reading of the leaderboard/label rules — hidden-label access banned; LB analysis and hypothesis-testing submissions OK; banned = iterative score-nudging (submit, tweak by .01, resubmit) and computing weights from scores"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-10-06T18:04:09.737Z
  modified_previous: 2026-10-05T14:56:03.405Z
---

Problem statement section 5 says "비공개 평가 정답에 접근하거나 이를 학습·특징 구성에 사용하는 행위는 금지". The user (team member deciding for the team) interprets this as a ban on obtaining the hidden label data itself (e.g. hacking the server, leaked files). Using the official leaderboard score feedback analytically — e.g. solving the four submissions' scores for segment-level bias, reading score changes as validation — is acceptable to them. Decided 2026-09-26.

Also decided 2026-10-01: public **train_y labels may be used as features for evaluation rows**. Official notice 2026-10-05 (중간안내 2절, original PDF in 공용/대회자료/공고/) confirms: all public train_X·train_y may be stored/referenced with no time restriction; time order = row_id relative day index + hour; eval rows may use only same-greenhouse current/earlier inputs; no later input at any step (incl. linking/correction); no fitting/updating on test inputs or their predictions; no adjusting to test-set statistics. Organizer add-on (6.344): comparing submissions by public score is fine; computing blend weights/correction coefficients from past test predictions + public scores is banned.

**Clarified 2026-10-07 (user):** the earlier "never use the leaderboard to choose directions" meant the user does NOT want **score-nudging loops** — submit, lower something by .01, submit again, repeat. Using a submission to test a genuinely different, pre-defined hypothesis/method (validated locally first, explainable) and comparing scores is acceptable.

**Why:** final presentation must explain why the model was built; "we tuned on the leaderboard" is not presentable, and organizers ban score-derived weights.

**Official clarification 2026-10-05 (organizer 중간 안내 PDF, catalog 6.309):** models trained on / storing and referencing ALL public train_X·train_y are allowed, and the time limit does NOT restrict training-data reference to before the eval row → later-day train_y labels may be referenced. Banned: any input later than the row at ANY step (features, preprocessing, search, linking, correction, post-processing); fitting/updating anything (incl. preprocessing such as a calendar) on test_X or its predictions; adjusting to test-set-wide statistics. Final answer invalid → latest of last 3 valid submissions that passes verification is used (not best score) → keep code/model/settings for every submission.

**Organizer 추가 안내 (user relayed 2026-10-05, catalog 6.344):** comparing submissions by public score is allowed; computing blend weights or correction coefficients from earlier evaluation-prediction files + public scores is NOT allowed (final answer invalid; fix before deadline). 25 submissions are for trying ideas, not for learning hidden labels. → Leaderboard-derived numbers must never set a weight/coefficient/threshold; avoid even letting LB-inferred test composition steer method choice.

**How to apply (2026-10-05 판):** leaderboard-score analysis is allowed for aggregate/segment-level diagnosis, not for building or selecting the submitted model. Still avoid back-solving individual test labels and hard-coding them into answers, since section 8.2 reproduction review reads the code. Related: [[analysis-before-answers]]

**How to apply (2026-10-07 갱신, 위 판과 함께 적용):** each submission = a distinct, locally validated, pre-specified method; report its score as evidence. Never fit weights/thresholds/coefficients to scores, never do small incremental nudges between submissions, never back-solve individual test labels. Related: [[analysis-before-answers]], [[skill-not-luck]]
