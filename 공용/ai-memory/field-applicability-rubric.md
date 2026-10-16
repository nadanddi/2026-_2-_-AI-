---
name: field-applicability-rubric
description: "Organizer presentation rubric (shared 2026-10-13) — 10 of 30 points for field applicability, which penalises reliance on data-collection artifacts"
metadata:
  node_type: memory
  type: project
  originSessionId: 738a1055-284b-445f-a192-fb89b6a6eba7
  modified: 2026-10-14T10:57:19.440Z
---

On 2026-10-13 the user shared the organizers' presentation rubric for the top-10 teams.

**Scoring:**
- The presentation is worth 30 points, of which 10 are "현장 적용 가능성" (field applicability): 5 for the structured mission and 5 for the image mission.
- 5 points: works with information available at prediction time, and the verification evidence shows no reliance on competition-specific regularities.
- 2 points: some components rely on such regularities, but the team separates them and reports performance and impact without them.
- 0 points: the core performance relies on those regularities with no evidence they hold in practice, or the presentation does not match the submitted code.
- Full PDF read 2026-10-14 (발표평가_현장적용가능성_안내_20261012.pdf): the 0-point line says "(분리 보고 여부와 무관)" — separating and reporting does NOT rescue a submission whose CORE performance relies on the regularities. "의존 정도와 현장 적용 근거가 우선, 한계를 공개했다는 사실만으로 점수를 주지 않음"; high online score or future plans do not prove applicability. Online legality is judged separately (시간 정보 사용 범위, 평가 입력 사용 범위, 자료 출처, 재현). So plan A (keep artifact modules) risks 0, not 2, when the artifact modules carry most of the gain.
- 2026-10-14 LB: round 15 gains (temp V3 GRU, EC HR5 TabPFN day-level corrector) came from GENERAL methods (own-record inputs only) — first evidence that general methods move the LB.

Online-mission legality is judged separately and is unchanged.

**Why it matters:** many of our gains rely on artifacts of how the competition data was built:
- record shuffling and reconstruction: linker anchors, W2 "number −2", DMT calendars;
- labelled training days interleaved with evaluation days, including next-day labels;
- outdoor twins across houses.

At real prediction time, future labels, and often past labels of an unmeasured target, are not available.

**How to apply:**
- Keep artifact-dependent parts as separable on/off modules.
- Always be able to report "base model only" against "with artifact modules". On the leaderboard that comparison is round 9 (temp .5127 / EC .1384) against rounds 12–13 (temp .4883 / EC .1271).
- When proposing new methods, label them general or artifact-dependent.
- Don't create presentation material unless asked ([[no-presentation-prep]]).
- Related: [[rule-safety-checklist]].

**User leaning (2026-10-13): plan A.**
- Keep the online-best configuration, including the artifact modules.
- Aim for 2 field points by separating those modules and reporting performance without them.
- Plan B (drop the artifact modules for 5 points) was judged riskier. Our image score is 22.94/30, roughly 9th, so the structured rank decides whether we reach the presentation round.
- If B is revisited, first get the leaderboard neighbours' temp and EC RMSE, and audit the base model (SG2, DP1, season index) for artifact dependence.

**User boundary (2026-10-13): "최대 규칙성 의존까지만 하자".**
- Allowed: artifact-dependent methods that use only past-in-time information, e.g. previous-day labels (anchors v7/v15/v12), W2, W2C, and same-ID earlier records.
- Excluded: anything using future information that would not exist at prediction time, e.g. next-day labels (BRK forward side, NXL, CAS's two-sided agreement step).
- Supporting evidence may still cite future-side data, but predictions must not use it.
