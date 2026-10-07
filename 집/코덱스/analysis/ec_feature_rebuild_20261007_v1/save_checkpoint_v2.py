"""Record actual CPU baseline progress without declaring goal completion."""
from pathlib import Path
import json,re,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r3=HERE/'checkpoints/BLK_R3_v1'
reg=read(r3/'registration.json')
assert read(r3/'complete.json')['status']=='R3_RAW_MEMBERS_COMPLETE'
members=[]
for seed in [47,1414,6464]:
    for name in ['ET','LGB','MLP']:
        p=r3/f'{name}_seed{seed}.json';r=read(p)
        assert r['registration_sha256']==sha(r3/'registration.json') and len(r['pred'])==1440
        assert not r['heldout_truth_loaded']
        members.append({'file':str(p.relative_to(HERE)),'sha256':sha(p)})
sg=read(HERE/'BLK_SG2_refonly_audit_v2.json');assert sg['status']=='PASS'
boundary=read(HERE/'BLK_endpoint_boundary_audit_v2.json');assert boundary['status']=='PASS'
assert not (HERE/'checkpoint_record_v2.json').exists()
record={'goal_complete':False,'new_candidate_performance_tested':False,'R3_members_complete':9,
    'whole_baseline_complete':False,'R3_member_files':members,'SG2_nonconstant_source_max_difference':sg['independent_source_max_difference'],
    'SG2_future_other_farm_checks':sg['future_other_farm_checks'],'endpoint_independent_formula_checks':boundary['independent_formula_checks'],
    'PFN_live_handle_last_verified_by_parent':2936,'PFN_live_PID_last_verified_by_parent':20072,
    'PFN_progress':'context5 running, contexts6..8 pending at recording; poll original handle before restart',
    'next':['poll PFN handle2936','close independent critique of v2 fixes','PFN broader final query invariance audit','assemble .6R3+.4PFN one shrink/clip/ref-only SG2','full-pipeline causal audit','BLK six variant diagnosis','original TM/P2LOO/EL1 safe endpoint selector before adoption']}
(HERE/'checkpoint_record_v2.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
p=HERE/'PROGRESS.md'
old=p.read_text(encoding='utf-8')
latest='''## 最新 실행 체크포인트 (2026-10-07, v2)

- 목표 미완료, 후보 성능 채점/채택/제출0. 아래 이전 체크포인트의 "own worker 없음"은 당시 상태이며 현재 상태가 아니다.
- CPU R3 ET/LGB/MLP × seed47/1414/6464 실제9fit exit0, train5520/query1440. checkpoints/BLK_R3_v1 실제orderedID/원본·의존SHA·raw출력checksum 및독립비평 확인. 평가정답 미로드.
- CPU TabPFN v2 문맥5..8 실제실행 시작. 마지막 확인 tool handle2936 / PID20072 live, context5 계산 중. 가중치SHA2ab5…고정·CPU·외부통신차단. 완료파일/잠금만으로live판정하지 말고 원handle을 먼저poll한다. 중복실행금지.
- SG2 ref-only adapter v2: pass1 query통계fit 위험 수정, 완전0..h prediction prefix·실제pivotcolumn순서 검사. 시간별비상수 예측 source대조1440행 max2.22e-16, future/otherfarm48 PASS. 전체PFN/R3 혼합결과 인과감사와 edge-case firstrecord/no-finite 별도시험은 남음.
- endpoint v2: 양끝정확ID/24h/immutable 참조/최종clip, 독립원ID시간 보간1440 PASS. 등록 v2에 baseline→endpoint 순서·정확loss·farm층화8block row-weightbootstrap·6variant 보정 고정.
- EC 연결은 실제 출처/방향 사슬을 입증하지 않은 유사성 연결성분. 크기·cycle·거리·guard coverage 진단 저장 필요. 기존TM/P2LOO/EL1 anchor규칙 미봉인이므로 현재 BLK만 효과진단가능, 전체채택금지.
- 재개: 원PFN handle 확인 → 최신비평 닫힘 확인 → 전체baseline 조합·단계/인과감사 → BLK 효과진단 → 원검증기 endpoint 규칙·교차검증. 도메인24→문헌24→나머지142 단계는 이후 계속.

'''.replace('最新','최신')
p.write_text(old.split('\n',1)[0]+'\n\n'+latest+old.split('\n',1)[1],encoding='utf-8')
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
text=catalog.read_text(encoding='utf-8')
numbers=[int(x) for x in re.findall(r'^\| 6\.(\d+) \|',text,re.M)]
number=max(numbers)+1
row=f'\n| 6.{number} | **BLK CPU 기준선 R3 9구성원 실행 및 SG2 query 통계 혼입 위험 수정, 후보 성능 미채점** (2026-10-07 집 코덱스) | train5520/query1440, ET/LGB/MLP×3seed raw체크포인트 exit0·독립checksum/ID/의존SHA 확인. BLK pass1 query를SG2 prepare에합치면pass1표준화통계fit에섞임→ref-only fit/prefix query 분리v2. 비상수시각예측 SG2 원함수대조max2.22e-16/future48 PASS, endpoint독립보간1440/immutable PASS. CPU PFN문맥5..8 계산진행,전체혼합/후처리 인과성 미완료. EC 연결성분은실제동사슬증명아님,기존3검증기anchor정의미봉인→BLK진단만허용·전체채택금지. 외부수치/GPU/검증정답채점/제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/checkpoint_record_v2.json;BLK_SG2_refonly_audit_v2.json;BLK_method_registration_v2.json;critique_BLK_baseline_adapter_v1.md |\n'
with catalog.open('a',encoding='utf-8') as f:f.write(row)
for target,body in [
    (ROOT/'집/코덱스/작업일지/2026-10-07.md',f'\n## BLK CPU 기준선 실행 및 SG2 분리 (6.{number})\n- R3 9개실제fit완료·정답미채점. SG2ref-only 비상수원대조/future검사PASS,endpointv2독립보간/불변성PASS. CPU PFN5..8 실행중,원handle2936/PID20072 마지막live확인;중복실행금지. 전체baseline/후보효과/TM·P2LOO·EL1anchor정의남음. PROGRESS/checkpoint_record_v2.json 재개.\n'),
    (ROOT/'공용/HANDOFF.md',f'\n### 2026-10-07 집 코덱스 BLK CPU 기준선 진행 (6.{number})\n- R3raw9개완료. SG2 BLKquery통계혼입방지ref-only adapterv2/source대조PASS. CPU PFN문맥5..8 실행중(마지막확인handle2936/PID20072),GPU0·이웃PF1/PF2중복0. 성능채점/후보채택/제출0. ec_feature_rebuild_20261007_v1/PROGRESS.md 최신체크포인트부터재개;원handle확인없이재시작금지.\n')]:
    with target.open('a',encoding='utf-8') as f:f.write(body)
print(json.dumps({'catalog_number':f'6.{number}','R3_members_complete':9,'goal_complete':False},ensure_ascii=False))
