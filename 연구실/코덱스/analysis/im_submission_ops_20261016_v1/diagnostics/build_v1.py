import csv,json,math,hashlib
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
DATA=ROOT/'연구실/코덱스/local/im_submission_ops_20261016_v1'
prev=ROOT/'연구실/코덱스/local/im_n9_error_diagnosis_20261016_v1/rows_v1.csv'
with prev.open(encoding='utf-8-sig') as f: raw=list(csv.DictReader(f))
assert hashlib.sha256(prev.read_bytes()).hexdigest()==json.loads((DATA/'prior_rows_sha_v1.json').read_text())['sha256']
with (ROOT/'연구실/코덱스/analysis/im_n9_error_diagnosis_20261016_v1/진단표_v2.csv').open(encoding='utf-8-sig') as f: prior=list(csv.DictReader(f)); cols=list(prior[0])
cols+=['원본후보발견','발견조건','마스크사유']
masks={(r['dataset'],r['file']):r for r in json.loads((DATA/'miss_masks_v1.json').read_text(encoding='utf-8'))}
miss=[r for r in raw if r['dataset']!='im_set_realg' and r['label']=='1' and float(r['p9'])<.5]
M=1.0110872983932495; GAP=3.4739584922790527
for r in miss:
    r['found']=r['dg1']!='' and r['gap']!='' and math.isfinite(float(r['dg1'])) and math.isfinite(float(r['gap'])) and float(r['dg1'])<=M and float(r['gap'])>=GAP
out=[]
for ds in ['전체','im_set_r2','im_set_r3','im_set_r4','im_set_r5']:
    group=[r for r in miss if ds=='전체' or r['dataset']==ds]
    for kind in ['전체']+sorted({r['kind'] for r in group}):
        gg=[r for r in group if kind=='전체' or r['kind']==kind]; n=sum(r['found'] for r in gg)
        out.append({'구분':'D1 유형별 원본후보 발견률','자료':ds.replace('im_set_',''),'항목':'발견 수/가짜 놓침 수','kind':kind,'표본수':n,'분모':len(gg),'값':n/len(gg),'발견조건':f'finite ff, dg1≤{M}, gap≥{GAP}','근거/주의':'고정 나단9 원본후보 규칙; 저장 ff에는 원본ID가 없어 실제 동일 원본 확인 아님'})
for r in miss:
    mask=masks[r['dataset'],r['file']]
    out.append({'구분':'D1 놓친 가짜 상세','자료':r['dataset'].replace('im_set_',''),'항목':'개별 사진','오류유형':'가짜 놓침','kind':r['kind'],'파일':r['file'],'원본':r['source'],'표본수':1,'p9':r['p9'],'게이트':r['gate'],'rc':r['rc'],'dg1':r['dg1'] or 'NA','gap':r['gap'] or 'NA','원본후보발견':int(r['found']),'마스크비율':'NA' if mask['area'] is None else mask['area'],'마스크평균알파':'NA' if mask['mean_alpha'] is None else mask['mean_alpha'],'마스크사유':mask['reason'],'근거/주의':'마스크는 저장된 마스크상의 픽셀 면적; 미보관은 0 아님'})
out.extend(dict(r,구분='D2 '+r['구분']) for r in prior if r['구분'].startswith('②'))
with (HERE/'진단표_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(out)
n=sum(r['found'] for r in miss); rate=n/len(miss); branch='A1' if rate>=.3 else 'A2'
res=dict(missed_fake=len(miss),found=n,rate=rate,branch=branch,m=M,gap=GAP,rc_count=sum(int(r['rc']) for r in miss),
         by_kind={k:dict(n=sum(r['kind']==k for r in miss),found=sum(r['found'] and r['kind']==k for r in miss)) for k in sorted({r['kind'] for r in miss})},table_rows=len(out))
with (HERE/'d1_result_v1.json').open('x',encoding='utf-8') as f: json.dump(res,f,ensure_ascii=False,indent=2)
lines=[f'1. D1의 R2~R5 가짜 놓침 {len(miss)}장 중 고정 원본후보 규칙을 통과한 것은 {n}장({rate:.2%}); 사전 30% 분기에 따라 {branch}를 진행합니다(진단 자체 채택 판정 없음).',
       f'2. 발견은 dg1≤{M}·gap≥{GAP}로 정의하며 rc와 별개입니다; 원본ID가 저장되지 않아 실제 동일 원본을 확인한 비율은 아닙니다.',
       '3. 놓침 전행의 kind·p·dg1·gap·rc·마스크 면적을 표에 수록했고, R2 등 저장 마스크가 없는 경우 NA와 사유를 표시했습니다.',
       '4. D2는 기존 고정 진단을 그대로 재사용했습니다: 고확신 진짜 오판 40행 대 정답 진짜 2,938행, 밝기 중앙값120.397/125.940·채도.267667/.287596·갈변대리.027024/.026691, 농가·품종·ff·rc 분포와 분모 포함.',
       '5. R2~R5는 이미 개봉한 탐색 자료이며 화면통계·원본후보 발견은 인과나 숨김평가 대표성 증거가 아닙니다; 모델·문턱·계수 변경은 이 진단에서 하지 않았습니다.']
with (HERE/'요약_v1.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print(json.dumps(res,ensure_ascii=False))
