# T-S2 기준 구성원의 잔차 트리에 계절 좌표 적용 · 2026-10-03

- T-S1은 전12칸 개선이지만 DIAG p .15655~.1655라 기각. 결과를 본 뒤 확장하는 순차 실험임을 명시한다.
- T-S2는 T-S1과 합치지 않는다. BASE의 LGB 잔차 트리에만 day->season 적용. 선형 물리식/Ridge/Nystroem/CODEX/PFN/혼합비율/게이트 고정.
- BASE는 기존 0.65res+0.25ridge+0.1nys를 재학습해 cache 일치를 확인. BASE시드7/101, CODEX726/727 짝, PFN1..8/17..24. 기존 DIAG10/EXT10/EXT12 그대로.
- season fit은 각 fold 학습일 날씨만, 검증 입력/정답 사용0. baseline cache maxdiff<1e-8 아니면 중단.
- 두 적용 위치를 시험했으므로 Bonferroni k=2: 12칸 전부 개선 + DIAG 각 p_worse<.0125 및97.5% delta MSE CI상한<0. 20,000회 온실5일블록/RNG20261003 고정. T-S1도 이 강화 기준에서는 기각 유지.
- 기존 캐시 재사용의 한계와 전체 학습 입력 기반 기존 가중치 rank는 T-S1과 동일. 제출/EC잠금 읽기 없음. 둘 모두 실패하면 현재 적용 실험 종료·W30G 유지.
