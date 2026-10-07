from pathlib import Path
import csv
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def load(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
with (HERE/'feature_candidates_v2.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS']:
    for i,method in enumerate(['PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD'],1):
        row={k:'' for k in rows[0]}
        row.update(candidate_id=f'BLK_{scope}_{i}',stage='0',family=method,source='사용자 추가 최우선 목표: 공개 학습 정답 양끝·사슬',
            previous_catalog='6.345;6.346;6.348;6.350',
            difference_from_previous='학습 정답 양끝 고정, gap 입력·정답 차단, 앞쪽 query 입력만 허용하는 새 BLK 배치',
            status='WAIT_BASELINE_AND_METHOD_SPEC_REVIEW',performance='UNTESTED',
            parameters=json.dumps({'baseline_scope':scope,'day_metadata':'원본 유지, 가짜 pass2 일차 금지',
                'SG2_scope':'모든 고정 BLK query에 역할로 적용' if scope=='BLK_QUERY_ROLE' else '배포 원본 day<179 gate 유지',
                'priority':'BLK_QUERY_ROLE 주 비교, BLK_RAW_PASS 적용범위 진단; 결과 보고 기준선 선택 금지',
                'method_spec':'끝점·사슬 점수/불확실성/fallback 세부 수식 봉인 전 fit 금지',
                'candidate_limit':'3개 방법×2범위=6variant. 두 범위 선별 시 k=6',
                'numeric_label_use':'공개 train_ids 내 정답만; query/gap/locked±1 양정답 금지'},ensure_ascii=False))
        rows.append(row)
assert len(rows)==len({r['candidate_id'] for r in rows})==196
with (HERE/'feature_candidates_v3.csv').open('x',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
feature=load('feature_audit_v1.json');fast=load('fast_feature_audit_v2.json')
blk=load('BLK_layout_v2.json');context=load('BLK_context_audit_v1.json');probe=load('baseline_probe_result_v1.json')
assert feature['all_checks_pass'] and context['status']=='PASS' and probe['status']=='PASS'
assert feature['rows']==fast['rows']==9600 and feature['columns']==fast['columns']==838
assert len(blk['query_ids'])==context['query_rows']==sum(len(b['query_days'])*24 for b in blk['blocks'])==1440
assert len(blk['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])==context['gap_rows_absent']==len(blk['blocks'])*2*24==384
assert len(blk['train_ids'])==context['train_rows']==5520
assert not(set(blk['train_ids'])&set(blk['hidden_both_label_ids']))
assert not(set(blk['train_ids'])&set(blk['locked_near_ids']))
progress='''# EC 변수 전수 재점검 — 재개 지점 (2026-10-07 집 코덱스)

목표 미완료. 신규 특징 성능/채택/최종 구성/제출 없음.

## 지금 최우선

사용자 추가 목표 BLK 양끝 정답 연결을 먼저 수행한다. 도메인①→문헌②→데이터③ 순서는 그 후 유지한다.
1. critique_BLK_baseline_scope_v1.md를 먼저 읽고 BLK 기준선·방법 수식을 봉인한다.
2. 주 비교 BLK_QUERY_ROLE: 원본 온실/일차/시각과 학습 달력은 유지, SG2 적용을 가짜 평가 query 역할로 명시한다.
3. BLK_RAW_PASS: 배포 day<179 gate를 그대로 둔 적용범위 진단이다. 원본 일차를 가짜 pass2로 변경하지 않는다.
4. CPU로 계절v2+DP1+PFN .4+SG2 기준선을 BLK에 재현하고 전체 처리 인과성을 확인한다. ET 한 셀 재현과 전체 기준선 재현을 구분한다.
5. PAST_ENDPOINT/BOTH_ENDPOINT/CHAIN_PREFIX_GUARD 세 방법의 점수·연결·모호할 때 fallback을 비평받고 고정 후 실행한다. 두 범위를 선별에 쓰면 6variant로 보정한다.
6. 블록 앞/중간/뒤, 일반/고EC일, 시드3개 이상을 비교한다. 좋으면 BLK 주검증기 후보로 제안하되 TM/P2LOO/EL1을 자동 폐기하지 않는다.

## 완료한 기초 작업

- 원자료 SHA/EC·온도 정답 수 재확인. 카탈로그 6.1~6.370의 388항목(중복 번호 포함) 스냅샷·전체 제목 목록, 관련 원문·구현 대조. 카탈로그 모든 주장 독립 검증 완료라는 뜻은 아니다.
- 후보 v3: BLK6 + 도메인24 + 문헌 대기24 + 나머지142 =196개 등록. 원열 전체 비공집합16383 및 상호작용/파생군 전체는 후속 배치 대기.
- 9600행×838 입력 기반 시간별 특징. v1 prefix/미래/순서/다른 온실/시간 누락/자정 reset 감사 PASS. 빠른 v2는40대조 PASS, 원방법 독립 수치대조는2일 한정.
- CPU ET 한 셀: 학습6288/검증912/47특징 seed47, 원 저장출력 최대차2.22e-16 PASS. LGB/MLP/PFN/SG2 전체 재현은 미완료.
- checkpoint v1 16오염거부, v2 단일writer claim·partial 보존·hash hex·PID 시작시각 합성 검사 PASS. 실제 장기 runner crash/동시process/stale PID 복구 미완료.
- 실제 원본·의존11해시 재검산 PASS. fit 직전 강제검사 연결은 남음.
- BLK v1은 역사 locked±1 누락 때문에 무효·미학습. v2는8덩어리, query1440/gap384/train5520, endpoint8쌍, raw gap 값 숫자 파싱 전 제거, 양정답 숨김.
- BLK 문맥1560검사 및 독립 비평 재검사 PASS. 완전 pass2 모양 후보0, 선택8개 모두pass1. 이 대표성 한계를 숨기지 않는다.
- 집 클로드 PF1/PF2 결과파일은 아직 완료본 없음. GPU 실행/이웃 중복 학습 금지. 결과 수신 시 출처·ID·인과성·성능 산술을 독립 감사한다.

## 재개 규칙

- 먼저 PROGRESS.md → BLK_layout_v2.json → feature_candidates_v3.csv → preregistration_v2.json → critic_fixes_v3.json 및 최신 비평을 읽는다.
- BLK layout v2 고정, 숫자 정답으로 배치를 바꾸지 않는다. gap은 입력·EC·온도 모두 제거하고 사슬/달력/표준화/검색에도 전달하지 않는다.
- blk_context_v1.BLKContext의 reference_inputs/reference_labels/query_prefix 경계로 구현한다. 뒤 query 입력을 직접 CSV에서 읽는 우회 코드는 금지한다.
- checkpoint receipt와 실제 원본/코드/환경/학습·참조·문맥·query/후처리 해시를 대조한다. 파일 존재만으로 재사용하지 않는다.
- 현재 실제 실행 중인 own worker 없음. baseline probe/slow feature audit/fast audit 정상 종료0. 과거 PID40500·tool76791을 live라고 오인하지 않는다.
- 처음 쓰는 확인 seed/layout와 최종 후보 k는 아직 봉인 전이다. 새로운 seed/layout도 새로운 정답은 아니다.
- 기존 결과는 보존하고 새 버전으로 추가한다. PROGRESS.md만 사용자 요청대로 갱신한다.
'''
(HERE/'PROGRESS.md').write_text(progress,encoding='utf-8')
report='''# EC 기초 재점검 중간 체크포인트 — 2026-10-07

아직 예측 효과 순위나 완료 보고서가 아니다. BLK 추가 목표가 최우선이다.

원입력14열 외에 동일 온실 현재·이전 입력, 공개 학습 EC/온도 정답 참조, row_id·쌍·순서·달력, 전체 온실 보조학습 후보를 등록했다. 공개 온도304965행 중 EC 없는 온도 정답295365행을 빠뜨리지 않는다. 현재 평가 실제 온도는 주어지지 않으므로 직접 입력하지 않는다.

BLK는 구조만으로8덩어리(각 온실5일2개·10일2개)를 고정했다. 가짜 평가1440행, 입력·두 정답 차단gap384행, 학습5520행이다. 역사잠금±1을 보호하고, 양끝23시/0시anchor8쌍을 봉인했다. 문맥1560검사와 독립재검사 PASS이다. 앞뒤 정답 사슬 효과는 아직 학습/검증하지 않았다.

완전한 pass2 구간에서는 이 모양을 만들 수 없어 현재 BLK는전부pass1이다. 배포SG2의pass1skip과WT2전체TM적용은같은baseline이아니다. 원일차를가짜pass2로바꾸지않고BLK_QUERY_ROLE주비교/BLK_RAW_PASS범위진단으로구분한다. 이때좋은결과가나와도실제pass2개선이나기존실패원인의확정증거로해석하지않는다.

시간 흐름 특징은9600행×838열 구현감사를통과했다. ET기준선한셀6288학습/912검증/47특징CPU재현최대차2.22e-16을독립검산했다. 이것은전체모델재현이나새피처개선이아니다.

독립비평지적과부분수정은critic_fixes_v3.json에남았다. 전체baseline·최종처리누수·후보별열명세·실제runner재개·최종새배치확인은남아있다. 후보196개모두효과미검증상태이며, 문헌24개는자료읽기·중복검토후활성화할대기슬롯이다.

다음재개는PROGRESS.md부터읽는다. 모든성과를기존기준선대비RMSE·시드/검증기·일반/고EC·블록위치별로정리하는완료보고서는실험뒤작성한다.
'''
p=HERE/'중간_체크포인트_v1.md';assert not p.exists();p.write_text(report,encoding='utf-8')
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
old=catalog.read_text(encoding='utf-8-sig');nums=[int(v) for v in re.findall(r'^\|\s*6\.(\d+)\s*\|',old,re.M)]
number=max(nums)+1
entry=f'\n| 6.{number} | **EC 기초 재점검 및 BLK 구조·문맥 등록/감사, 성능 미시험** (2026-10-07 집 코덱스) | 원자료 온도304965/EC9600 재검산·후보196등록. 시간별 입력9600×838의 prefix/미래/순서/타농장/시간누락/자정reset PASS, CPU ET한셀6288학습/912query/47특징seed47 저장출력max2.22e-16 독립일치. BLK v1 locked±1 누락발견·fit전v2수정,8덩어리query1440/gap384/train5520,양끝anchor8쌍·gap 입력/양정답 차단·문맥1560검사 및독립PASS. 완전pass2모양0/선택전부pass1;배포SG2 pass1skip 대 WT2전체TM 적용차이를 BLK_RAW_PASS/BLK_QUERY_ROLE 사전분리 필요. 전체기준선·사슬효과·후보RMSE·최종채택 미검증, 새후보학습/제출0. 원자료 입력 밖 외부수치 사용0·GPU0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/PROGRESS.md;feature_audit_v1.json;baseline_probe_result_v1.json;BLK_layout_v2.json;BLK_context_audit_v1.json;critique_BLK_baseline_scope_v1.md |\n'
# Re-read immediately before append; do not reuse an old last-number assumption.
assert catalog.read_text(encoding='utf-8-sig')==old,'catalog changed, rerun with a new output version'
with catalog.open('a',encoding='utf-8') as handle:handle.write(entry)
note=f'\n## EC 기초 재점검 새 목표 및 BLK 최우선\n- 카탈로그6.{number}, 결과물 ec_feature_rebuild_20261007_v1/만 생성. PROGRESS.md 재개지점·후보196·비평/수정/근거 연결. 도메인24/문헌대기24/나머지142+BLK6.\n- BLK8덩어리 query1440/gap384/train5520, locked±1 보호·gap 입력과양정답숫자파싱전차단·양끝8쌍·문맥1560+독립PASS. pass2구조불가/pass1만. 원일차변조금지/SG2 적용범위두baseline구분.\n- 시간별838특징9600행 입력인과PASS. ET기준선한셀재현PASS, 전체모델/사슬개선/후보채택0. 실행own worker모두exit0·재시작대상없음. PF1/PF2 GPU와이웃중복0.\n- 다음:PROGRESS부터 BLK 기준선·3방법 수식봉인/CPU실행→TM/P2LOO/EL1 유지·새확인1회, 이후도메인→문헌→자료순.\n'
with (ROOT/'집/코덱스/작업일지/2026-10-07.md').open('a',encoding='utf-8') as handle:handle.write(note)
with (ROOT/'공용/HANDOFF.md').open('a',encoding='utf-8') as handle:handle.write('\n### 2026-10-07 집 코덱스 EC 기초 재점검 및 BLK 최우선 재개\n'+note)
out=HERE/'checkpoint_record_v1.json';assert not out.exists()
out.write_text(json.dumps({'catalog_number':f'6.{number}','registered_candidates':len(rows),
    'feature_audit_checks':len(feature['checks']),'BLK_context_checks':context['query_checks_and_future_other_farm_perturbations'],
    'goal_complete':False,'new_candidate_performance_tested':False,'own_workers_running':False,
    'next':'BLK_QUERY_ROLE baseline and 3 endpoint-chain method specification/review/CPU execution'},ensure_ascii=False,indent=2),encoding='utf-8')
print(out.read_text(encoding='utf-8'))
