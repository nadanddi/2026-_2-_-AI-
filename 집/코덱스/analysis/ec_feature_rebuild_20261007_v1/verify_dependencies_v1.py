from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
contract=json.loads((HERE/'source_contract_v1.json').read_text(encoding='utf-8'))
checks=[]
for name,expected in contract['SHA256'].items():
    path=ROOT/'공용/대회자료/정형데이터/참가자_배포'/name
    actual=sha(path);assert actual==expected
    checks.append({'path':str(path),'SHA256':actual,'matches_registration':True})
probe=json.loads((HERE/'baseline_probe_registration_v1.json').read_text(encoding='utf-8'))
for relative,expected in probe['source_hashes'].items():
    actual=sha(ROOT/relative);assert actual==expected
    checks.append({'path':relative,'SHA256':actual,'matches_probe':True})
for path in [HERE/'checkpoint_v1.py',ROOT/'집/클로드/research/env.py',ROOT/'연구실/클로드/code/common.py']:
    checks.append({'path':str(path),'SHA256':sha(path),'pin_status':'새 실행 전 다시 대조해야 함'})
out=HERE/'dependency_recheck_v1.json';assert not out.exists()
out.write_text(json.dumps({'status':'PASS','checks':checks,'limitation':'현재 파일 해시 재검산. runner가 fit 직전 이 검사를 실행하도록 연결하는 작업은 남음'},ensure_ascii=False,indent=2),encoding='utf-8')
fixes=[
 {'issue':'C01','status':'DOCUMENTED','evidence':'feature_candidates_v2.csv; 고정 family 묶음 정의','remaining':'생성 ordered columns와 variant 명세 고정'},
 {'issue':'C02','status':'PARTIAL','evidence':'보조·구조8후보 등록, 상호작용 문법 명시','remaining':'상호작용/파생군/조건부 후보 후속 배치'},
 {'issue':'C03','status':'DOCUMENTED','evidence':'preregistration_v2.json full_path_ablation','remaining':'간접경로 source lineage와 달력 순환 감사'},
 {'issue':'C04','status':'PARTIAL','evidence':'baseline_probe_registration_v1.json 실제6288/912 ID·달력 fit ID·PFN 잠정문맥','remaining':'전체66fold 및 후보별 inner/온도 보류 계약'},
 {'issue':'C05','status':'PARTIAL','evidence':'ET 구성원 실제 CPU 재현 PASS, 독립 비평v2','remaining':'다른 구성원/혼합/SG2 재현 및 전체 인과성'},
 {'issue':'C06','status':'DOCUMENTED','evidence':'preregistration_v2.json 새seed/fold!=새정답','remaining':'최종seed/layout/finalists/k 봉인 및 노출 이력'},
 {'issue':'C07','status':'PENDING','evidence':'문헌24슬롯 WAIT 상태','remaining':'문헌별 읽기 범위/URL/실측대응/중복원코드 비교'},
 {'issue':'C08','status':'RUNNING','evidence':'causal_features_v1.py/audit_features_v1.py 실제CPU session76791','remaining':'최종 감사 파일/후보별 경계 명세'},
 {'issue':'C09','status':'PARTIAL','evidence':'checkpoint_v1.py 16개 오염거부 PASS','remaining':'동시writer/crash/stale PID/실제runner 재개'},
 {'issue':'V201','status':'FIXED_DOCUMENT','evidence':'이 파일이 원비평 C01~C09 매핑을 바로잡음. 이전 매핑 v2 보존'},
 {'issue':'V202','status':'PARTIAL','evidence':'dependency_recheck_v1.json 원본·원모듈 실제 SHA 재계산','remaining':'fit 전 강제호출/전체 의존성 pin'},
 {'issue':'V204','status':'PENDING','evidence':'개선fit gate 닫음','remaining':'후보별 정확한 열집합·계약·runner'},
 {'issue':'V205','status':'PENDING','evidence':'v1 checkpoint는 합성 단일writer 검증만 통과','remaining':'exclusive writer·부분결과 보존·PID시작시각·hex검사'},
]
out=HERE/'critic_fixes_v3.json';assert not out.exists()
out.write_text(json.dumps(fixes,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','hashes_recomputed':len(checks),'formal_candidate_fit_allowed':False}))
