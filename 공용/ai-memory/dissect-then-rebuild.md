---
name: dissect-then-rebuild
description: Don't reuse earlier models/parts wholesale — dissect them, keep only the truly useful mechanism, rebuild it for our current setup, and check the 5-point field rubric
metadata:
  type: feedback
---

The user said (2026-10-15): "이런식으로 이전 모델에서 도움이 될만한 부분 그대로 쓰지 말고 안에 세부적으로 뜯어봐서 정말로 살릴 부분만 빼가고 우리쪽에 도움 되도록 수정하는 방식으로 해야해. 대신 발표 5점 규정에 들어가야하니까 그건 꼭 확인하고."

**Why:** Re-running old parts as-is (AG6 mixture, FLN anchor swap, KEA, YL1 features) kept failing or giving ~0; the CMX1 dissection (break-even precision ~.8, same-date neighbours uninformative, AG6 relied on calendar structure) explained why and pointed to what is salvageable.

**How to apply:** For any earlier experiment worth revisiting:
1. Decompose its mechanism into parts and measure which part carries the gain, and on which segment.
2. Classify each part F/R/X under the 5-point rubric. Drop X; R only with a field-valid restoration.
3. Rebuild just the useful part for the current configuration (FL linker, NX base, round-27 structure).
4. Then run a pre-registered test.
Related: [[skill-not-luck]], [[field-applicability-rubric]], [[check-catalog-before-new]].
