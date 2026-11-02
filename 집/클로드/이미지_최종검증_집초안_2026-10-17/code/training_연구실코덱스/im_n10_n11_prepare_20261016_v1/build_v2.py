"""Create independent N9-derived packages, cached regression checks and ZIPs."""
import sys,json,csv,shutil,hashlib,zipfile,ast
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;B=Path.home()/'storage/nadan/farmai_img';SRC=B/'submit_img09/팜모니_나단9'
RAW=BASE/'im_ff_rule_diagnosis_20261016_v1/raw_v2';GAP=3.4739584922790527;DG=1.0911458730697632
source=(SRC/'inference.py').read_text(encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
end='    return min(max(p, 0.0), 1.0)'
assert source.count(end)==1
alarm='''    # N10: fixed E3 distance rule, independent of the JPEG expert gate.
    ff = info.get("ff")
    if ff is not None and len(ff) == 5 and np.isfinite(ff).all():
        if ff[1] >= 3.4739584922790527 and ff[0] > 1.0911458730697632:
            p = max(p, 0.6)
'''
cond=' and p < 0.8: p = min(p, 0.3); rc = True'
assert source.count(cond)==1
texts={10:source.replace(end,alarm+end),11:source.replace(cond,': p = min(p, 0.3); rc = True')}
before={str(p.relative_to(SRC)):sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
result={}
for num in (10,11):
    root=HERE/'build_v2'/f'n{num}';team=root/f'팜모니_나단{num}';root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(SRC,team,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    ast.parse(texts[num]);(team/'inference.py').write_text(texts[num],encoding='utf-8')
    readme=f'''# 팜모니_나단{num}: 가설 검증 제출 후보

나단9에서 {'ff유효·gap>=3.4739584922790527·dg1>1.0911458730697632이면 최종p=max(p,.6),게이트무관' if num==10 else '진짜확정 ppre<.8 조건만 제거,가짜우선분기·기존전문가유지'}.
학습·모델·DB·기타문턱 변경 없음. 다른후보와 혼합하지 않음.

가설 검증 제출, 효과는 시험 사진 원본이 AI Hub DB에 있을 때만, 위험 = 운영진 진짜 재처리 시 오판.
위 문구는 기대효과의 가정이며 원본 존재를 인증한 것이 아니다. 원본이 DB에 없어도 최근접 지문 규칙이 발동해 오탐할 수 있다.
DB출처는 기존나단9README와같은AI Hub공개원본152043장의정밀지문이며원본사진/정답포함안함.
기존나단9문서와재현설명은팀원용설명문참조. E4시험입력통계는미사용.
'''
    # Preserve the source README inside the explanation/repro bundle, outside model package.
    (root/'README_n9_original.md').write_bytes((SRC/'README.md').read_bytes())
    (team/'README.md').write_text(readme,encoding='utf-8')
    assert all(sha(team/rel)==s for rel,s in before.items() if rel not in ('inference.py','README.md'))
    size=sum(p.stat().st_size for p in team.rglob('*') if p.is_file());assert size<1_000_000_000
    metrics={}
    for p in sorted(RAW.glob('*.npz')):
        with np.load(p) as z:
            y=z['y'];base=z['p9'];ff=z['ff'];alarm_ix=np.isfinite(ff).all(1)&(ff[:,1]>=GAP)&(ff[:,0]>DG)
            candidate=np.maximum(base,np.where(alarm_ix,.6,0)) if num==10 else z['p9_no_pcap']
            assert np.array_equal(alarm_ix,z['valid']&(ff[:,1]>=GAP)&(ff[:,0]>=np.nextafter(DG,np.inf)))
            ba=lambda a:float(.5*((a[y==0]<.5).mean()+(a[y==1]>=.5).mean())) if (y==0).any() and (y==1).any() else None
            fixed=int(((base>=.5)!=y).sum()-((candidate>=.5)!=y).sum())
            metrics[p.stem]=dict(n=len(y),base_ba=ba(base),candidate_ba=ba(candidate),base_correct=int(((base>=.5)==y).sum()),candidate_correct=int(((candidate>=.5)==y).sum()),net_fixed=fixed,new_fp=int(((y==0)&(base<.5)&(candidate>=.5)).sum()),fixed_fake=int(((y==1)&(base<.5)&(candidate>=.5)).sum()),new_fn=int(((y==1)&(base>=.5)&(candidate<.5)).sum()))
            np.savez(root/f'expected_{p.stem}.npz',files=z['files'],y=y,expected=candidate,base=base)
    assert metrics['pub21']['candidate_correct']==21
    if num==10:
        with (BASE/'im_ff_rule_diagnosis_20261016_v1/진단표_v1.csv').open(encoding='utf-8-sig') as f:tab=list(csv.DictReader(f))
        ref={r['자료']:r for r in tab if r['구분']=='E3 자료별경보' and r['집단']=='ff유효_gap기존이상' and r['항목']=='dg1'}
        for sn,v in metrics.items():assert abs(v['candidate_ba']-float(ref[sn]['반사실BA']))<1e-15 if v['candidate_ba'] is not None else True
    archive=root/f'팜모니_나단{num}.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as zz:
        for p in sorted(team.rglob('*')):
            if p.is_file():zz.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as zz:assert zz.testzip() is None
    result[str(num)]=dict(metrics=metrics,size=size,zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),source_model_unchanged=True,changes=['inference.py','README.md'])
    print('BUILT',num,size,archive.stat().st_size,flush=True)
assert before=={str(p.relative_to(SRC)):sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
with (HERE/'build_result_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('BOTH_BUILT',flush=True)
