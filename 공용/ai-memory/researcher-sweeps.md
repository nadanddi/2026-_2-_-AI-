---
name: researcher-sweeps
description: Don't test one conventional value — sweep each weight/threshold/learning setting over a reasoned range, justify every formula, select on one split and judge on another
metadata:
  type: feedback
---

User (2026-10-12): "적당한 크기, 관례적인 퍼센트로 한 번만 실험하지 말고 연구자의 입장에서 어느 정도 다양하게 실험을 해보고 평가를 정립하도록 해. 연구자는 모든 계산 식이나 검증에 이유가 있잖아."

**Why:** many earlier constants were picked once by convention (0.5 shrink, 0.4 PFN, w 1.0, clip ±1, 0–7 h window, p ≥ .8). A single conventional value can sit far from the optimum, or the effect can depend on it in ways we never saw.

**How to apply:**
- For every new mechanism, write down WHY each parameter exists and what physical or statistical quantity sets its scale. Examples: decay time from the residual autocorrelation, bias from the 23 h → 0 h cooling.
- Sweep a grid over a sensible range. Show the response curve or surface.
- Choose on one split (pass-1 rows or inner folds) and judge on another (pass-2, mirror CV). Report the whole curve, not only the best point. Count how many settings were tried, and treat a lone sharp optimum as suspect.
- Measure the parameter from data when possible, instead of tuning it.
- Still pre-register the selection rule (which split chooses, which judges) before running. Related: [[skill-not-luck]], [[harsh-critic-every-result]].
