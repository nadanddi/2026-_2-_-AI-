from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,hashlib,shutil,collections
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name;DEST=LOCAL/'유사사진_통합104장';DEST.mkdir(exist_ok=False)
old=json.loads((RUN.parent/'image_similar_collection_20261010_v1/final_manifest_v1.json').read_text(encoding='utf-8'))['images']
new=json.loads((RUN/'final_manifest_v1.json').read_text(encoding='utf-8'))['rows']
rows=[];seen=set();cache=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file());own=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file());added=0
for phase,rr in [('1차49',old),('추가55',new)]:
    for r in rr:
        raw=Path(r['final_path']).read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==r['sha256'] and sha not in seen;seen.add(sha)
        assert cache+added+len(raw)<16*1024**3 and own+added+len(raw)<2*1024**3 and shutil.disk_usage(ROOT).free-len(raw)>30*1024**3
        p=DEST/(r['source_archive'].removesuffix('.tar')+'__'+Path(r['source_member']).name)
        with p.open('xb') as f:f.write(raw)
        added+=len(raw)
        if phase=='1차49':
            refs=r['visual_references'];kind=r['kind'];farm=r['farm'];date=r['date'];prefix=r['prefix']
        else:
            v=r['visual_review'];refs=[dict(public_file=v['reference'],degree=v['degree'],reason=v['reason'])];kind=r['kind_type'];farm=r['farm_id'];date=r['date_captured'];prefix=r['prefix_candidate']
        rows.append(dict(file=p.name,path=str(p),phase=phase,image_id=r['image_id'],kind=kind,farm=farm,prefix=prefix,date=date,reference='; '.join(z['public_file'] for z in refs),degree='; '.join(z['public_file']+': '+z['degree'] for z in refs),reason='; '.join(z['public_file']+': '+z['reason'] for z in refs),sha256=sha,bytes=len(raw),source_archive=r['source_archive'],source_member=r['source_member']))
assert len(rows)==len(seen)==104
with (DEST/'사진별_선정이유_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['파일명','선정차수','ID','품종','농장','그룹접두사','촬영일시','공개참조','유사강도','선정이유','SHA256'])
    for r in rows:w.writerow([r['file'],r['phase'],r['image_id'],r['kind'],r['farm'],r['prefix'],r['date'],r['reference'],r['degree'],r['reason'],r['sha256']])
with (DEST/'먼저_읽기_v1.txt').open('x',encoding='utf-8') as f:f.write('AI허브에서 선별한 원본 JPEG104장: 기존49+추가55. 금실83/설향21. 원본무편집,사진별공개참조·유사강도·이유는CSV.\n104는완전동일픽셀중복이없는파일수이며독립식물/장면수는아님.\n검색은기존VS1000+개발6농장15TAR의추가408장. 전체350GB전수검색은아님.\n사진모음의유사강도는정성판정이며,학습효용/동일공개원본판정은아님. 공개근접계보의추가415753 등은향후독립검증에주의. 원본촬영메타데이터의농장/품종은별도실물확증아님.\n')
out=dict(count=len(rows),rows=rows,folder=str(DEST),kind_counts=dict(collections.Counter(r['kind'] for r in rows)),farm_counts=dict(collections.Counter(r['farm'] for r in rows)),bytes=added)
with (RUN/'combined_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k] for k in ['count','folder','kind_counts','farm_counts','bytes']},ensure_ascii=False))
