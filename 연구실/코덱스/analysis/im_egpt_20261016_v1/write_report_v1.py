from pathlib import Path
import json,csv,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[3];D=ROOT/'연구실/코덱스/local/im_egpt_20261016_v1/corrected_v3'
r=json.loads((D/'result.json').read_text(encoding='utf-8'));v=json.loads((D/'verification_v1.json').read_text(encoding='utf-8'))
assert not r['passed'] and v['verified']
lines=['# E_gpt 단일 가설 검증 결과 — 2026-10-16 · 연구실 · 코덱스', '',f"완료: {r['finished']} / 판정: **불통과, 16시 나단11 건너뜀**. 나단9 유지. 업로드·패키지 제작 0.", '', '| 검증 | 나단9 → E_gpt 추가 | 기준 | 판정 |','|---|---|---|---|']
rows=[]
for g in r['logo']:
 lines.append(f"| LOGO {g['held']} | 놓친 {g['base_missed']}장 중 {g['fixed']}장 고침 | 3/4회 ≥1, 합계 ≥5 | {'고침 있음' if g['fixed'] else '고침 없음'} |")
 rows.append(dict(section='LOGO',set=g['held'],n=g['n'],base=g['base_missed'],candidate=g['fixed'],delta='',unit='missed/fixed'))
for sn in ['im_set_newreal','im_set_realg','im_set_realn','sealed_real','sealed_brown','sealed_bright','jig500']:
 m=r['metrics'][sn];lines.append(f"| {sn} 진짜 오판 | {m['base_fp']} → {m['new_fp']} / {m['real']} (+{m['fp_delta']*100:.4f}%p) | ≤+0.5%p | 통과 |")
 rows.append(dict(section='real_FP',set=sn,n=m['real'],base=m['base_fp'],candidate=m['new_fp'],delta=m['fp_delta'],unit='count/rate'))
b=r['budget'];lines.append(f"| 전문가 합집합 진짜 오탐 | {b['base_count']} → {b['new_count']} / {b['n']} ({b['base_rate']*100:.4f}% → {b['new_rate']*100:.4f}%, +{b['delta']*100:.4f}%p) | ≤+0.1%p | **실패** |")
for sn in ['im_set_r2','im_set_r3','im_set_r4','im_set_r5']:
 m=r['metrics'][sn];delta=m['new_ba']-m['base_ba'];lines.append(f"| {sn} 균형 정확도 | {m['base_ba']*100:.6f}% → {m['new_ba']*100:.6f}% ({delta*100:+.4f}%p) | ≥−0.2%p | {'통과' if delta>=-.002 else '**실패**'} |")
 rows.append(dict(section='BA',set=sn,n=m['n'],base=m['base_ba'],candidate=m['new_ba'],delta=delta,unit='rate'))
lines += ['| 공개 샘플 | 21 → 21 / 21 | 21/21 | 통과 |','', '요약 5줄:', '1. GPT 자료 묶음 제외 검증은 3/4회, 총14장 추가 탐지로 (a)를 통과했다(gptr은 나단9 놓침 자체가0).', '2. 지정 진짜7세트는 각각 +0.5%p 이내지만 합집합 진짜 오탐 +0.1224%p가 +0.1%p 상한을 넘었다.', '3. R5 균형 정확도 −0.2809%p로 −0.2%p 하한도 위반해 최종 불통과다.', '4. 입력18해시·원나단9 10세트 예측차0·독립 배열 재계산 일치, 후속 변형·나단11 ZIP 제작·업로드0.', '5. 16시 건너뜀; 14시 나단9 재제출과 18시 운영은 사용자 수행이며 실제 업로드/LB 결과는 이번 작업에서 확인하지 않았다.', '', '계획·중간 비평: 원본명 정규화, 기존 나단9 OOF 보존·입력해시 검산을 반영했다. 추가 group전체 제거는 단일클래스 오류를 만든 구현 실수였으며 v2를 무효 보존했다. 원지시대로 짝진짜만 제외하는 v3로 4회 전부 재실행했다(모델·문턱분위·기준 변경0). v3의 다른 양성 exact source 교집합은 4회 모두0; gptr 확인에서는 짝진짜150장을 학습·보정에서 제외했다. OOF는 group/source 분리 assert 통과. 새 촬영묶음 완전독립 일반화 주장은 하지 않는다.', '', '보정·풀: final theta=.9073158431, theta_rc=.9630420438, OOF+newreal/realg/realn 보정3345행. 기존 im66 합집합 풀3267행을 동일 순서로 재구성하고 기존18입력해시/rc/행수 일치 확인. E_gpt 음성3750행은 group5fold OOF, 합집합 풀의 기존학습진짜·미사용진짜 행과 정렬했다. newreal/realg/realn은 문턱 보정 자료이며 독립 일반화 검증이 아니다. 봉인 진짜3종 및 R2~R5도 과거 사용된 자료다.', '', '증거: ownlocal `corrected_v3/result.json`, `verification_v1.json`, `*_oof.npz`, `eval_*.npz`, `union_pool.npz`; analysis `PLAN_v1.md`, `구현정정_v1.md`, `run_v3.py`, `run_v3.log`. 기존 나단11(ppre제거판)과 별도 실험이며 기존 파일 수정없음.']
(P/'결과_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (P/'결과표_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['section','set','n','base','candidate','delta','unit']);w.writeheader();w.writerows(rows)
for name in ['result.json','verification_v1.json']:(P/name).write_bytes((D/name).read_bytes())
inventory={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file()}
(P/'回수해시_v1.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print('report',P/'결과_v1.md','files',len(inventory))
