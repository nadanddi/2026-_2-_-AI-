"""Write a new report from frozen results; no models/data re-scoring."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
r=json.loads((H/'results_v1.json').read_text(encoding='utf-8'));a=json.loads((H/'feature_audit_v2.json').read_text(encoding='utf-8'));v=json.loads((H/'independent_verify_v3.json').read_text(encoding='utf-8'))
lines=['# 温度モデル検討' if False else '# 온도 모델 품질 보완·CO₂ 구성원 진단 결과','',
'원 제출 모델은 보존했다. 검증 시 학습 가중치와 계절 변환을 outer 훈련 입력에만 맞추고, 전문가 답변을 기존 관측과 연결했다. 아래 성능은 새 환경의 CODEX 구성원 비교이며 9회차 W40G-S 전체 모델의 성능이나 리더보드 개선이 아니다.','',
'## 바꾼 것','',
'- CV 학습 가중치의 CO₂ rank·cold EWMA는 해당 fold 훈련 입력만으로 계산한다. 기준/후보 모두 같은 수정 적용. 계절 fit도 같은 훈련 날씨만 사용한다.',
'- 후보 DROP5는 CO₂ 관련 다섯 잔차 입력을 삭제한다. MED3은 같은 기록일 현재와 과거 2시간, 세 시간칸의 중앙값으로 CO₂를 평활하고 그 다섯 열만 다시 만든다. 물리 Ridge·BASE/PFN·구동기 act_co2 변경 없음.',
'- CODEX는 현 설정에서 실질 결정적이므로 seed 9109 한 번으로 구성원 진단. 같은 모델을 여러 seed로 반복한 것을 독립 확인이라고 세지 않았다.',
'- 후보2개와 모든 판단조건을 fit 전에 등록했다. 모든 DIAG10/EXT10/EXT12 방향 개선 + DIAG p_worse<.0125 + 중앙97.5% CI 상단<0일 때만 전체 모델 검증에 진입하도록 했다. 구성원 단계 통과도 전체 채택은 아니다.','',
'## 실제 결과','',
'| 검증기 | 후보 | 기준 RMSE(℃) | 후보 RMSE(℃) | 변화 |','|---|---|---:|---:|---:|']
for s,ms in r['results'].items():
    for c,m in ms.items():lines.append(f"| {s} | {c} | {m['base_rmse']:.6f} | {m['candidate_rmse']:.6f} | {m['pct']:+.3f}% |")
lines+=['','변화는 음수일 때 개선이다.']
for c in ['DROP5','MED3']:
    m=r['results']['DIAG10'][c];lines.append(f"- {c}: DIAG p_worse={m['p_worse']:.5f}, CI[{m['ci'][0]:+.6f}, {m['ci'][1]:+.6f}]℃. 전체 검증 진행 기준 {'통과' if r['screen_pass'][c] else '미달'}.")
lines+=['','## 범위와 판정','',f"- 총 구성원 fit {r['total_models']}회(기준/두 후보 × 12fold). 첫fold 3회 후 중간 비평, 남은 33회 완료. 새로운 baseline/candidate만 사용했으며 TK/CUDA/옛 W30G 혼합 cache를 정확 기준으로 재사용하지 않았다.",
f"- 특징/가중치 경계 검사 {a['cells']:,}셀 PASS. 원 가중치1=7259행/.2=2318행/.04=23행. 원 source/data SHA 전후 유지. 이 검사만으로 전체 PFN8 모델 예측의 규정 검사를 완료했다고 말하지 않는다.",
f"- 독립 산술·ID·purge·계절 anchor·bootstrap 검산 {v['checks']:,}건 PASS, 최대 수치차 {v['max_abs_numeric_gap']:.3g}. EC 정답 수치 사용0, 외부 자료 사용0, 새 제출/공식평가예측0.",
'- 판단 기준 미달이면 비용 정책상 이번 후보의 전체 혼합 검증에 진입하지 않는다. 이 결정은 전체 혼합에서 효과가 전혀 없다는 증명이 아니다.',
'- 기존 DIAG10을 여러 번 본 탐색 진단이며 독립 잠금/신선한 최종 확인으로 부르지 않는다. 새 모델 채택0. 소수6자리 원 제출 재현 실패(6.474)는 여전히 원인 미확정이다.','',
'## 전문가 설명 반영','',
'원 질문지 PDF 2쪽·답변txt·대응정리를 읽었다. 전문가가 배지와 공기 온도의 분리가 정상일 수 있다고 설명한 것은 날 삭제/공기온도 강제 수축의 근거가 아니다. 멀칭·센서 위치·관수 수온은 가능한 원인이나 실제 기록에서 확인된 원인이 아니다. 물 수온·멀칭 여부·실제 동 ID를 임의로 만들지 않는다. 질문지의 약0.8℃ 차이도 모든 행에 보장된 보정식이 아니다.',
'자세한 근거 연결은 전문가_온도반영_v2.md. CODEX에는 이미 온도/일사/난방 1·3·8시간 필터가 있고 BASE에는 커튼-유효일사가 있으므로, 다음에 검토할 것은 기존 BASE의 유효일사와 지연을 CODEX 잔차로 옮기는 좁은 후보다. 그 후보는 아직 등록·실행하지 않았다.','',
'## 비평과 보완','',
'계획 비평의 시간격자·fold-MASK·CI 정의를 고정했다. 중간 비평에서 검사 자체가 매일 오후를 바꾸는 오류를 고쳐 전체 시간축의 단일 절단점으로 검사했다. 첫fold의 파일/ID/purge/가중치/유한 예측은 독립 검산 완료. 비평가가 등록source불일치라고 추정한 지적은 실제 SHA 직접 확인으로 철회(MID_REVIEW_v3). 최종 비평 결과는 FINAL_REVIEW_v1.md에 별도 기록한다.','']
with (H/'보고서_v1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
print('\n'.join(lines[:26]))
