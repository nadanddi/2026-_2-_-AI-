# 제1회 전국 농과계 대학 농업 AI 경진대회 — 정형데이터 미션

딸기 온실의 환경·구동기 기록으로 같은 시간대의 **배지 온도(`sub_temp`)** 와 **배지 EC(`sub_ec`)** 를 추정합니다.
팀 **팜모니** · 마감 2026-10-16 18:00 · 제출 25회

---

## 어디부터 봐야 하나

| 알고 싶은 것 | 보는 곳 |
|---|---|
| **지금 상황이 어떤가** | [`docs/현황.md`](docs/현황.md) |
| **무엇이 바뀌었나 (최근 작업)** | [`docs/변경이력/`](docs/변경이력/) |
| **데이터에서 뭘 발견했나** | [`research/데이터_단서_카탈로그.md`](research/데이터_단서_카탈로그.md) |
| **제출한 모델이 뭐였나** | [`research/submissions/설명자료_*.md`](research/submissions/) |
| **작업을 이어받으려면** | [`HANDOFF.md`](HANDOFF.md) → [`docs/문서지도.md`](docs/문서지도.md) |
| **환경 설정 / 두 기기 동기화** | [`docs/연구실PC_설정_안내.md`](docs/연구실PC_설정_안내.md) |

처음 오셨다면 **`docs/현황.md` → `research/데이터_단서_카탈로그.md`** 순서를 권합니다.

---

## 폴더 구조

```
docs/                문서 전용. 코드 없음
  현황.md              순위·제출이력·다음 할 일 (한 장 요약)
  문서지도.md           모든 문서가 어디 있는지
  변경이력/             ★ AI·사람이 작업 후 여기에 기록을 남깁니다
  ai-memory/           AI 작업 규칙 (한국어 사용, 근거 우선 등)
  연구실PC_설정_안내.md   환경 구축·동기화

research/            현재 주력 작업 (Codex 담당: EC)
  데이터_단서_카탈로그.md  ★ 가장 중요한 문서. 모든 발견이 번호로 정리됨
  제출기록.md
  submissions/         제출 파일과 회차별 모델 설명
  local/               무거운 결과물 (git 제외, 드라이브 동기화)
  *.py                 실험 스크립트

Claude/              Claude 담당 작업 (온도)
  code/                실험·제출 스크립트
  logs/                모든 실험 실행 기록 = 수치의 근거
  results/             실험 결과 CSV
  submissions/         제출본
  HANDOFF.md           Claude 계열 인계 문서

analysis/            초기 탐색 및 독립 검증 (Codex v1~v5, 감사)
  codex_independent/   가설별 실험 폴더. 각 폴더에 PROTOCOL.md + 결과보고서.md

blind/               블라인드 교차검증 (A_opus / B_sonnet / C_haiku)
tools/               환경 설정·동기화 스크립트
정형데이터/            대회 원본 CSV
```

---

## 작업 규칙 (요약)

- **검증**: 평가 배치를 복제한 블록 CV + 다중 분할 평균. 단일 분할 결과로 판단하지 않습니다.
- **규정**: 같은 온실의 현재·과거 입력만 사용. `causality_test` 계열로 자동 검사합니다.
- **기록**: 실험을 하면 결과를 **카탈로그에 번호로 추가**하고, 작업 단위가 끝나면 **`docs/변경이력/`에 한 장** 남깁니다.
- 자세한 규칙은 [`CLAUDE.md`](CLAUDE.md) · [`AGENTS.md`](AGENTS.md)

## 브랜치

| 브랜치 | 용도 |
|---|---|
| `codex-ec` | 현재 주력 (최신) |
| `feature/rmse-improvement` | 온도 개선 계열 |
| `claude-handoff` | Claude 인계분 |
| `main` | 초기 스냅샷 |
