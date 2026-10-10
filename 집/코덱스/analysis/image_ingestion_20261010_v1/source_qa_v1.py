import sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,csv,hashlib,shutil,collections
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent
LOCAL=ROOT/'집'/'코덱스'/'local'/'image_ingestion_20261010_v1'
SOURCE=Path(r'G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터\1.Training\원천데이터_add_20260807')
CAP=2*1024**3;IMAGE_CAP=20*1024**2

def octal(b):
    s=b.strip(b'\0 ')
    if not s:return 0
    if any(c not in b'01234567' for c in s):raise ValueError('non-octal')
    return int(s,8)

def index_tar(p):
    entries={};names=set();pos=0;length=p.stat().st_size
    with p.open('rb') as f:
        while pos+512<=length:
            f.seek(pos);h=f.read(512)
            if h==bytes(512):
                if pos+1024>length:raise ValueError('missing end blocks')
                while True:
                    tail=f.read(1024**2)
                    if not tail:break
                    if any(tail):raise ValueError('nonzero tail')
                if not entries:raise ValueError('empty TAR')
                return entries
            if len(h)!=512 or sum(h[:148]+b' '*8+h[156:])!=octal(h[148:156]):raise ValueError('checksum')
            kind=h[156:157]
            if kind not in [b'0',b'\0',b'5']:raise ValueError('unsupported type')
            name=h[:100].split(b'\0',1)[0].decode('utf-8')
            if h[257:263] in [b'ustar\0',b'ustar ']:
                prefix=h[345:500].split(b'\0',1)[0].decode('utf-8')
                if prefix:name=prefix+'/'+name
            name=name.replace('\\','/');pp=PurePosixPath(name)
            if pp.is_absolute() or '..' in pp.parts or ':' in name:raise ValueError('unsafe')
            normalized=str(pp)
            if normalized in names:raise ValueError('duplicate path')
            names.add(normalized);size=octal(h[124:136]);end=pos+512+((size+511)//512)*512
            if end>length-1024:raise ValueError('extent')
            if kind!=b'5':
                if pp.suffix.lower() not in ['.jpg','.jpeg','.png']:raise ValueError('unexpected image entry')
                if pp.name in entries:raise ValueError('ambiguous basename')
                entries[pp.name]={'member':normalized,'offset':pos+512,'bytes':size}
            pos=end
    raise ValueError('missing end')

def tree_bytes(p):return sum(q.stat().st_size for q in p.rglob('*') if q.is_file()) if p.exists() else 0

def main():
    out=RUN/'source_qa_v1.json'
    if out.exists():raise RuntimeError('immutable output')
    metadata=list(csv.DictReader((RUN/'metadata_all_v1.csv').open(encoding='utf-8-sig')))
    selected=[];archives=[];copies=0;imagebytes=0;pixels=0;issues=[]
    for name in ['TS_1.금실12.tar','TS_2.설향10.tar']:
        p=SOURCE/name;st=p.stat()
        if st.st_size<=0 or copies+st.st_size+1>CAP or list(p.parent.glob(name+'.*')):raise ValueError('source readiness gate')
        current=tree_bytes(LOCAL)+tree_bytes(ROOT/'집'/'코덱스'/'local'/'image_runtime_20261010_v1')
        if current+st.st_size+IMAGE_CAP>10*1024**3 or shutil.disk_usage(ROOT).free-st.st_size<30*1024**3:raise ValueError('resource gate')
        target=LOCAL/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise RuntimeError('snapshot exists')
        sha=hashlib.sha256();copied=0
        with p.open('rb') as src,target.open('xb') as dst:
            while copied<st.st_size:
                chunk=src.read(min(8*1024**2,st.st_size-copied))
                if not chunk:raise ValueError('early EOF')
                dst.write(chunk);sha.update(chunk);copied+=len(chunk)
                if shutil.disk_usage(ROOT).free<30*1024**3:raise ValueError('space gate')
            if src.read(1):raise ValueError('unexpected trailing source')
        after=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('changed source')
        copies+=copied;idx=index_tar(target)
        tl=name.replace('TS_','TL_',1);rows=[r for r in metadata if r['archive']==tl]
        missing=[r['fname'] for r in rows if r['fname'] not in idx]
        if missing:raise ValueError('TL/TS mapping missing '+str(len(missing)))
        archives.append({'archive':name,'bytes':copied,'sha256':sha.hexdigest(),'image_members':len(idx),'TL_rows':len(rows),'mapping_missing':missing,'mapping_extra':len(set(idx)-{r['fname'] for r in rows}),'provider_completion':'unknown','scope':'verified stable local working snapshot'})
        ordered=sorted(rows,key=lambda r:(r['date_captured'],r['fname']))
        chosen=[ordered[round((len(ordered)-1)*q/3)] for q in range(4)]
        with target.open('rb') as f:
            for r in chosen:
                item=idx[r['fname']]
                if imagebytes+item['bytes']>IMAGE_CAP:
                    issues.append({'fname':r['fname'],'reason':'encoded total budget'});continue
                f.seek(item['offset']);raw=f.read(item['bytes'])
                import io
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width*im.height>25_000_000 or pixels+im.width*im.height>200_000_000:raise ValueError('pixel budget')
                    im.load();width,height=im.size;rgb=ImageOps.exif_transpose(im).convert('RGB');actual=rgb.size
                dest=LOCAL/'selected_real'/r['fname'];dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():raise RuntimeError('selected file exists')
                dest.write_bytes(raw);pixels+=width*height;imagebytes+=len(raw)
                mismatch=str(width)!=r['width'] or str(height)!=r['height']
                selected.append({**r,**item,'archive':name,'local_path':str(dest),'sha256':hashlib.sha256(raw).hexdigest(),'actual_width':width,'actual_height':height,'oriented_size':actual,'metadata_dimension_mismatch':mismatch,'scope':'train-only real QA; provenance binary label pending source documentation','selection':'pre-registered chronological four quantiles'})
        print(json.dumps({'archive':name,'copied_bytes':copied,'selected_count':len(selected)},ensure_ascii=False),flush=True)
    result={'archives':archives,'selected':selected,'direct_source_bytes':copies,'selected_encoded_bytes':imagebytes,'decoded_pixels':pixels,'issues':issues,'locked_farm_images_opened':0,'training':False,'model_inference':False}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
