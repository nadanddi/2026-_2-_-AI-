"""One diagnostic table, no selection/adoption decisions."""
import csv,json,re,random,zlib,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent; RUN=HERE.parent; ROOT=HERE.parents[4]
DATA=ROOT/'연구실/코덱스/local/im_n9_error_diagnosis_20261016_v1'
with (DATA/'rows_v1.csv').open(encoding='utf-8-sig') as f: rr=list(csv.DictReader(f))
def f(r,k): return float(r[k]) if r.get(k,'')!='' else np.nan
def canon_archive(v):
    v=Path(v.replace('\\','/')).name.removesuffix('.tar')
    return ('TL_'+v[3:]) if v.startswith('TS_') else ('VL_'+v[3:]) if v.startswith('VS_') else v
def base(v): return Path(v.replace('\\','/')).name
needed={(canon_archive(r['tar']),base(r['source'])) for r in rr}
index=defaultdict(set); aliases=Counter()
for p in (ROOT/'집/코덱스/analysis/image_ingestion_20261010_v1/metadata_all_v1.csv',ROOT/'집/코덱스/analysis/image_download_complete_20261010_v1/validation_metadata_v1.csv'):
    with p.open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            arc=canon_archive(r['archive']); member=Path(base(r['member'])).with_suffix('.jpg').name
            for nm in {base(r['fname']),member}:
                if (arc,nm) in needed:
                    index[arc,nm].add((r['farm_id'],r['kind_type'])); aliases[arc,nm]+=1
mapping_counts=Counter()
for r in rr:
    key=(canon_archive(r['tar']),base(r['source'])); matches=index[key]
    tarcv='금실' if '금실' in r['tar'] else '설향' if '설향' in r['tar'] else '미확인'
    mf=r['manifest_farm_id']
    if len(matches)==1:
        farm,cv=next(iter(matches))
        if (mf and mf!=farm) or (tarcv!='미확인' and cv!=tarcv):
            r.update(farm_id='충돌',cultivar='충돌',mapping='manifest/메타데이터 충돌')
        else: r.update(farm_id=farm,cultivar=cv,mapping='archive+fname/member 확정')
    elif len(matches)>1: r.update(farm_id='충돌',cultivar='충돌',mapping='메타데이터 다중농가 충돌')
    else: r.update(farm_id=mf or '미확인',cultivar=tarcv,mapping='manifest 농가/품종' if mf else '농가 미확인; tar 품종')
    mapping_counts[r['mapping']]+=1
notes={
'g_src2_014.jpg':'흰 멀칭과 두 단 재배대, 큰 녹색 잎·꽃, 후면 세로 창망이 보임. 병해·AI 여부는 사진만으로 판정하지 않음.',
'g_src2_139.jpg':'잎·줄기 근접 구도, 전체 세부가 부드럽고 노란 숫자 표찰·배지가 보임. 일부 작은 잎 끝 갈색점은 관찰에 한정.',
'g_src3_052.jpg':'강한 역광과 밝게 날아간 상단 창, 녹색 잎·꽃과 흰 멀칭의 명암 대비가 큼.',
'g_src3_081.jpg':'광택 있는 잎이 겹치고 중앙의 갈색 배지가 크게 보임. 갈변 대리값에는 배지 색도 포함될 수 있음.',
'g_src3_141.jpg':'잎 근접 구도와 검은 재배통·배지, 중앙 잎에 밝거나 갈색인 가는 흔적이 보임. 질감은 비교적 부드러움.'}
gpt=ROOT/'연구실/코덱스/local/ai_gen_work_20261014_v3'
gpts=sorted((gpt/'결과_GPT/3_부분생성').glob('*.png'))+sorted((gpt/'결과_GPT/4_부분스타일').glob('*.png'))+sorted((gpt/'중간_GPT').glob('*.png'))
longs=[]; pool=[]
for p in gpts:
    with Image.open(p) as im: size=im.size
    longs.append(max(size)); pool.append(dict(path=p.relative_to(gpt).as_posix(),width=size[0],height=size[1]))
assert longs
paths={}
for r in rr:
    if r['dataset']!='im_set_realg' or int(r['Ef_fire'])!=1: continue
    assert r['file'] in notes
    name=r['info'].split(';')[1]; rawdir='원본2' if name.startswith('src2_') else '원본3'
    raw=ROOT/'연구실/클로드/local/ai_gen_work'/rawdir/name
    with Image.open(raw) as im: w,h=im.size
    rng=random.Random(zlib.crc32(('G|'+name).encode())); target=rng.choice(longs)
    scale=target/max(w,h); nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    interp=rng.choice(['LANCZOS','BICUBIC'])
    tw,th=(768,1024) if nh>=nw else (1024,768)
    fac=max(tw/nw,th/nh); cw,ch=max(tw,round(nw*fac)),max(th,round(nh*fac))
    path=f'{w}×{h}→长边{target}({nw}×{nh},LANCZOS)→PNG无损往返→{interp}覆盖{cw}×{ch}→中心裁剪{tw}×{th}→JPEG Q92/420'
    path=path.replace('长边','긴 변').replace('无损往返','무손실 왕복').replace('覆盖','덮기').replace('中心裁剪','중앙 크롭')
    paths[r['file']]=dict(path=path,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),target_long=target,pool_n=len(longs),note=notes[r['file']])
cols=['구분','자료','비교집단','항목','오류유형','kind','파일','원본','표본수','분모','값','p9','p6','게이트','rc','마스크비율','마스크평균알파','밝기','채도','갈변대리비율','dg1','gap','ff유효수','ff결측수','농장코드','농가ID','품종','E_f','E_f문턱','발동전문가','처리경로','사진메모','근거/주의']
out=[]
def row(**kw):
    assert set(kw)<=set(cols); out.append(kw)
def indiv(r,panel,err):
    row(**{'구분':panel,'자료':r['dataset'].replace('im_set_',''),'비교집단':err,'항목':'개별 사진','오류유형':err,'kind':r['kind'],'파일':r['file'],'원본':r['source'],'표본수':1,
        'p9':r['p9'],'p6':r['p6'],'게이트':r['gate'],'rc':r['rc'],'마스크비율':r['mask_area'] or 'NA','마스크평균알파':r['mask_mean_alpha'] or 'NA',
        '밝기':r.get('brightness',''),'채도':r.get('saturation',''),'갈변대리비율':r.get('brown',''),'dg1':r['dg1'] or 'NA','gap':r['gap'] or 'NA',
        '농장코드':r['farm_code'],'농가ID':r['farm_id'],'품종':r['cultivar'],'E_f':r['Ef'],'E_f문턱':r['Ef_theta'],'발동전문가':r['expert_fires'] or '없음',
        '근거/주의':r['mapping']+('; 저장 마스크상 >127.5 면적; 원본 실제편집 면적과 구분' if r['mask_area'] else '')})
r5=[r for r in rr if r['dataset']=='im_set_r5']
for kind in sorted({r['kind'] for r in r5}):
    group=[r for r in r5 if r['kind']==kind]
    bad=[r for r in group if (f(r,'p9')>=.5)!=(r['label']=='1')]
    et='진짜 오판' if group[0]['label']=='0' else '가짜 놓침'
    row(**{'구분':'① R5 유형 집계','자료':'r5','항목':'오답 수/유형별 전체','오류유형':et,'kind':kind,'표본수':len(bad),'분모':len(group),'값':len(bad)/len(group)})
for r in r5:
    if (f(r,'p9')>=.5)!=(r['label']=='1'): indiv(r,'① R5 오답 상세','진짜 오판' if r['label']=='0' else '가짜 놓침')
core=[r for r in rr if r['dataset']!='im_set_realg' and r['label']=='0']
high=[r for r in core if f(r,'p9')>=.6]; correct=[r for r in core if f(r,'p9')<.5]
for r in high: indiv(r,'② 고확신 진짜 오판 상세','진짜 오판 p≥.6')
def stat(rows,k):
    x=np.array([f(r,k) for r in rows]); x=x[np.isfinite(x)]
    return 'NA' if len(x)==0 else f'{np.median(x):.6f} [{np.quantile(x,.25):.6f}, {np.quantile(x,.75):.6f}]'
metrics=[]
for ds in ('전체','im_set_r2','im_set_r3','im_set_r4','im_set_r5'):
    for groupname,group0 in (('진짜 오판 p≥.6',high),('정답 진짜 p<.5',correct)):
        gg=[r for r in group0 if ds=='전체' or r['dataset']==ds]; n=len(gg)
        valid={k:sum(np.isfinite(f(r,k)) for r in gg) for k in ('dg1','gap')}
        rec={'구분':'② 진짜 비교 통계','자료':ds.replace('im_set_',''),'비교집단':groupname,'항목':'중앙값 [Q25,Q75]·rc 비율','표본수':n,'분모':n,
             'rc':f'{sum(int(r["rc"]) for r in gg)}/{n}', '밝기':stat(gg,'brightness'),'채도':stat(gg,'saturation'),'갈변대리비율':stat(gg,'brown'),
             'dg1':stat(gg,'dg1'),'gap':stat(gg,'gap'),'ff유효수':f'dg1={valid["dg1"]}; gap={valid["gap"]}',
             'ff결측수':f'dg1={n-valid["dg1"]}; gap={n-valid["gap"]}', '근거/주의':'행 기준; 화면 전체 픽셀 통계; 개별사진 독립성/원인 판정 아님'}
        row(**rec); metrics.append(rec)
        for label,key in (('농가ID','farm_id'),('품종','cultivar'),('농장코드','farm_code')):
            counts=Counter(r[key] for r in gg)
            for name,num in sorted(counts.items()):
                row(**{'구분':'② 농가·품종 분포','자료':ds.replace('im_set_',''),'비교집단':groupname,'항목':label,'표본수':num,'분모':n,'값':num/n if n else 'NA',label:name,'근거/주의':'각 비교집단 전체 N이 분모; 미확인/충돌 포함'})
    total=[r for r in core if ds=='전체' or r['dataset']==ds]
    row(**{'구분':'② 비교 분모','자료':ds.replace('im_set_',''),'항목':'제외된 중간확률 진짜 오판 .5≤p<.6','표본수':sum(.5<=f(r,'p9')<.6 for r in total),'분모':len(total),'근거/주의':'고확신 오판/정답 진짜 두 비교집단 모두에서 제외'})
for r in rr:
    if r['dataset']=='im_set_realg' and r['Ef_fire']=='1':
        indiv(r,'③ realg E_f 발동 5장','진짜 E_f 발동'); rec=out[-1]
        rec['처리경로']=paths[r['file']]['path']; rec['사진메모']=notes[r['file']]
        rec['근거/주의']+='; im28 고정 RNG와 현재 보관 GPT 치수 pool 재생; 생성 당시 pool 동일성·JPEG byte 재현은 미인증; E_f 단독 원인 단정 안함'
definitions=[('밝기','풀해상도 RGB 평균(0~255)'),('채도','풀해상도 OpenCV RGB→HSV S 평균/255'),('갈변대리비율','R>G & G−B>8 & 50<(R+G+B)/3<200; 토양·시설 포함 가능, 병해 판정 아님'),('마스크비율','저장 마스크 >127.5 픽셀 비율; 마스크 없는 유형 NA; 원본 편집 면적 아님'),('ff·rc','기존 특징 캐시 dg1/gap·rc 사용; ff 결측은 NA와 유효/결측수 별도'),('메타데이터 결합','archive(TS→TL,VS→VL)+fname 또는 JSON member stem; 다중농가/품종 충돌은 충돌로 보존'),('G_path','AI허브 진짜→GPT 출력 치수 pool 기반 긴변조정→PNG 왕복→dist_cover→JPEG Q92/420; AI로 다시 그린 사진 아님')]
for name,meaning in definitions: row(**{'구분':'정의','항목':name,'근거/주의':meaning})
with (RUN/'진단표_v1.csv').open('x',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(out)
with (HERE/'build_audit_v1.json').open('x',encoding='utf-8') as f:
    json.dump(dict(table_rows=len(out),r5_errors=sum((f(r,'p9')>=.5)!=(r['label']=='1') for r in r5),high_fp=len(high),correct_real=len(correct),mapping_counts=dict(mapping_counts),metadata_ambiguous_keys=sum(len(v)>1 for v in index.values()),gpath=paths,pool=pool,summary_metrics=metrics),f,ensure_ascii=False,indent=2)
print('TABLE_BUILT',len(out),'high_fp',len(high),'correct_real',len(correct),dict(mapping_counts))
