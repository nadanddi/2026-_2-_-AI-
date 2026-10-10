import sys
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,hashlib,csv,collections,shutil,datetime
RUN=Path(__file__).resolve().parent
LOCAL=ROOT/'집'/'코덱스'/'local'/'image_ingestion_20261010_v1'/'label_snapshots'
DATA=Path(r'G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터\1.Training\라벨링데이터_add_20260807')
MAX=4*1024**2; TOTAL=600*1024**2
FIELDS=['farm_id','kind_type','crops_id','fname','date_captured','image_id','width','height','growth_stage']

def unique_pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('duplicate JSON key '+k)
        out[k]=v
    return out

def bad_constant(value): raise ValueError('nonfinite JSON '+value)

def octal(b):
    s=b.strip(b'\x00 ')
    if not s: return 0
    if any(c not in b'01234567' for c in s): raise ValueError('non-octal TAR field')
    return int(s,8)

def tar_records(raw):
    if len(raw)%512: raise ValueError('unaligned TAR')
    pos=0;seen=set();count=0
    while pos+512<=len(raw):
        header=raw[pos:pos+512]
        if header==bytes(512):
            if pos+1024>len(raw) or any(raw[pos:]): raise ValueError('invalid end padding')
            if count==0: raise ValueError('empty archive')
            return
        chk=octal(header[148:156]); altered=header[:148]+b' '*8+header[156:]
        if sum(altered)!=chk: raise ValueError('header checksum')
        kind=header[156:157]
        if kind not in [b'0',b'\x00',b'5']: raise ValueError('unsupported TAR extended/link/sparse type '+repr(kind))
        name=header[:100].split(b'\x00',1)[0].decode('utf-8')
        if header[257:263] in [b'ustar\x00',b'ustar ']:
            prefix=header[345:500].split(b'\x00',1)[0].decode('utf-8')
            if prefix: name=prefix+'/'+name
        name=name.replace('\\','/')
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name: raise ValueError('unsafe path')
        normalized=str(path)
        if normalized in seen: raise ValueError('normalized duplicate name')
        seen.add(normalized)
        size=octal(header[124:136]); end=pos+512+((size+511)//512)*512
        if end>len(raw)-1024: raise ValueError('member extent')
        if len(seen)>5000: raise ValueError('member count budget')
        if kind!=b'5':
            if not normalized.lower().endswith('.json') or size>1024**2: raise ValueError('unexpected member')
            payload=raw[pos+512:pos+512+size]
            obj=json.loads(payload.decode('utf-8-sig'),object_pairs_hook=unique_pairs,parse_constant=bad_constant)
            if not isinstance(obj,dict) or not isinstance(obj.get('images'),dict): raise ValueError('schema images')
            count+=1
            yield normalized,hashlib.sha256(payload).hexdigest(),obj
        pos=end
    raise ValueError('missing TAR end')

def main():
    output=RUN/'metadata_all_v1.csv'; report=RUN/'metadata_all_profile_v1.json'
    if output.exists() or report.exists(): raise RuntimeError('immutable outputs')
    LOCAL.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((RUN/'inventory_v1.json').read_text(encoding='utf-8'))
    paths=[Path(r['path']) for r in inventory['rows'] if r['kind']=='TL' and r['candidate']]
    read_bytes=0; archives=[]; rows=[]; issues=[];presence={k:collections.Counter() for k in FIELDS}
    for p in paths:
        try:
            st=p.stat()
            if st.st_size<=0 or st.st_size>=MAX: raise ValueError('size gate')
            if list(p.parent.glob(p.name+'.*')): raise ValueError('temporary companion')
            if read_bytes+st.st_size+1>TOTAL: raise ValueError('total read gate')
            if shutil.disk_usage(ROOT).free<30*1024**3: raise ValueError('free space gate')
            with p.open('rb') as f: raw=f.read(st.st_size+1)
            read_bytes+=len(raw); after=p.stat()
            if len(raw)!=st.st_size or (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('changed snapshot')
            parsed=list(tar_records(raw))
            staged=[]
            for member,sha,obj in parsed:
                im=obj['images'];row={k:im.get(k) for k in FIELDS}
                for k in FIELDS:
                    state='absent' if k not in im else 'null' if im[k] is None else 'blank' if im[k]=='' else 'present'
                    presence[k][state]+=1
                row.update(archive=p.name,member=member,json_sha256=sha,prefix_candidate=str(im.get('fname','')).rsplit('_',1)[0],prefix_meaning='unverified',licenses=json.dumps(obj.get('licenses'),ensure_ascii=False))
                staged.append(row)
            cache=LOCAL/p.name
            if cache.exists(): raise ValueError('cache exists')
            cache.write_bytes(raw)
            rows.extend(staged)
            archives.append({'archive':p.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'records':len(staged),'scope':'readable stable snapshot, provider completion unknown'})
        except Exception as e:
            issues.append({'archive':p.name,'error':str(e)})
        if len(archives)%25==0: print(json.dumps({'successful_archives':len(archives),'rows':len(rows),'read_bytes':read_bytes,'issues':len(issues)},ensure_ascii=False),flush=True)
    fieldnames=FIELDS+['archive','member','json_sha256','prefix_candidate','prefix_meaning','licenses']
    with output.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    cross=collections.Counter((str(r['farm_id']),str(r['kind_type'])) for r in rows)
    prefix=collections.Counter((str(r['farm_id']),str(r['kind_type']),r['prefix_candidate']) for r in rows)
    names=collections.Counter(str(r['fname']) for r in rows)
    dates=[r['date_captured'] for r in rows if isinstance(r['date_captured'],str) and r['date_captured']]
    result={'source_inventory_candidates':len(paths),'successful_archives':len(archives),'records':len(rows),'direct_read_bytes':read_bytes,'archives':archives,'issues':issues,'farm_cultivar_rows':[{'farm':k[0],'cultivar':k[1],'rows':v} for k,v in sorted(cross.items())],'prefix_candidates_count':len(prefix),'prefix_meaning':'not verified plant IDs','field_presence':presence,'duplicate_fname_excess':sum(n-1 for n in names.values() if n>1),'date_range_lexical':[min(dates),max(dates)] if dates else [],'snapshot_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'scope':'labels only; no image decode/visual QA; no farm-CV feasibility claim','annotations_bbox_checked':False}
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['archives','field_presence']},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
