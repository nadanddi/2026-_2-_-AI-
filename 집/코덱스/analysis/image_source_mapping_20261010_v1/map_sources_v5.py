import sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,hashlib,shutil,collections,unicodedata,argparse
RUN=Path(__file__).resolve().parent
LOCAL=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
PREV=RUN.parent/'image_download_complete_20261010_v1'
OLD=ROOT/'집/코덱스/local/image_ingestion_20261010_v1/source_snapshots'
SOURCES=['VS_1.금실2.tar','VS_2.설향3.tar']
ALLOWED={'AIF005','AIF006','AIF007'}
IO={'direct':0,'local':0}
def tracked_read(f,n,source=False):
    kind='direct' if source else 'local';limit=(2 if source else 10)*1024**3
    if n<0 or IO[kind]+n>limit:raise ValueError('read ledger budget')
    b=f.read(n);IO[kind]+=len(b);return b
def newbytes():return sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())
def save(name,obj):
    with (RUN/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def norm(s):return unicodedata.normalize('NFC',s).casefold()
def octal(b):
    s=b.strip(b'\x00 ')
    if any(c not in b'01234567' for c in s):raise ValueError('non-octal')
    return int(s,8) if s else 0
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:tracked_read(f,1024**2),b''):h.update(b)
    return h.hexdigest()
def ownbytes():
    return sum(p.stat().st_size for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1'] for p in (ROOT/'집/코덱스/local'/n).rglob('*') if p.is_file())
def index(p):
    total=p.stat().st_size
    if total%512:raise ValueError('alignment')
    pos=0;names=set();stems=set();result=[]
    with p.open('rb') as f:
        while pos+512<=total:
            f.seek(pos);h=tracked_read(f,512)
            if h==bytes(512):
                if total-pos<1024:raise ValueError('end padding')
                for b in iter(lambda:tracked_read(f,1024**2),b''):
                    if any(b):raise ValueError('nonzero trailing bytes')
                if not result:raise ValueError('empty')
                return result
            if sum(h[:148]+b' '*8+h[156:])!=octal(h[148:156]):raise ValueError('checksum')
            kind=h[156:157]
            if kind not in [b'0',b'\x00',b'5']:raise ValueError('unsupported TAR type')
            name=h[:100].split(b'\0',1)[0].decode('utf-8')
            if h[257:263] in [b'ustar\0',b'ustar ']:
                pre=h[345:500].split(b'\0',1)[0].decode('utf-8')
                if pre:name=pre+'/'+name
            name=name.replace('\\','/'); pp=PurePosixPath(name)
            if pp.is_absolute() or '..' in pp.parts or ':' in name:raise ValueError('unsafe path')
            key=norm(str(pp))
            if key in names:raise ValueError('duplicate path')
            names.add(key);size=octal(h[124:136]);end=pos+512+((size+511)//512)*512
            if end>total-1024:raise ValueError('extent')
            if kind==b'5' and size!=0:raise ValueError('nonempty directory')
            if kind!=b'5':
                stem=norm(pp.stem)
                if pp.suffix.lower() not in ['.jpg','.jpeg'] or size>32*1024**2 or size==0 or stem in stems:raise ValueError('unexpected image/duplicate stem')
                stems.add(stem);b=tracked_read(f,size)
                if len(b)!=size:raise ValueError('short payload')
                result.append({'member':str(pp),'stem':stem,'offset':pos+512,'bytes':size,'sha256':hashlib.sha256(b).hexdigest()})
            if len(names)>5000:raise ValueError('member budget')
            pos=end
    raise ValueError('missing end')
def mapping():
    inv=json.loads((PREV/'inventory_v1.json').read_text(encoding='utf-8'))
    labels=list(csv.DictReader((PREV/'validation_metadata_v1.csv').open(encoding='utf-8-sig')))
    LOCAL.mkdir(parents=True,exist_ok=True);archives=[];mapped=[];requested=0
    srcbytes=sum(x['bytes'] for x in inv['rows'] if x['name'] in SOURCES)
    oldbytes=sum(p.stat().st_size for p in OLD.glob('*.tar'))
    local_read_upper=2*srcbytes+2*oldbytes+64*32*1024**2+128*1024**2+16*1024**2
    if local_read_upper>10*1024**3:raise ValueError('local IO gate')
    reservation=json.loads((RUN.parent/'image_ingestion_20261010_v1/split_reservation_v1.json').read_text(encoding='utf-8'))
    manifest=[]
    for name in SOURCES:
        rr=[r for r in labels if r['archive']=='VL'+name[2:]]
        if len(rr)!=500 or any(r['farm_id'] not in ALLOWED or r['farm_id'] not in reservation['development_farm_folds'] for r in rr):raise ValueError('pre-read farm role')
        keys=[norm(PurePosixPath(r['member']).stem) for r in rr]
        if len(set(keys))!=len(rr) or any(json.loads(r['invalid_identity_fields']) or norm(PurePosixPath(r['member']).stem).rsplit('_',1)[-1]!=str(r['image_id']) for r in rr):raise ValueError('pre-read identity')
        manifest.append({'source_archive':name,'label_archive':'VL'+name[2:],'rows':len(rr),'farms':dict(collections.Counter(r['farm_id'] for r in rr))})
    save('preflight_v1.json',{'manifest':manifest,'direct_request_upper':srcbytes+len(SOURCES),'local_read_upper':local_read_upper,'cache_budget':12*1024**3,'new_budget':2*1024**3,'free_reserve':30*1024**3})
    for name in SOURCES:
        item=next(x for x in inv['rows'] if x['name']==name);p=Path(item['path']);st=p.stat();q=LOCAL/name
        if q.exists():raise ValueError('immutable snapshot')
        if (st.st_size,st.st_mtime_ns)!=(item['bytes'],item['mtime_ns']) or list(p.parent.glob(name+'.*')):raise ValueError('source change/temp')
        requested+=st.st_size+1
        if newbytes()+st.st_size+128*1024**2>2*1024**3:raise ValueError('new cache budget')
        if requested>2*1024**3 or ownbytes()+st.st_size+128*1024**2>12*1024**3 or shutil.disk_usage(ROOT).free-st.st_size<30*1024**3:raise ValueError('budget')
        h=hashlib.sha256();read=0
        with p.open('rb') as src,q.open('xb') as dst:
            while True:
                b=tracked_read(src,min(1024**2,st.st_size-read+1),source=True)
                if not b:break
                read+=len(b)
                if read>st.st_size:raise ValueError('source grew')
                dst.write(b);h.update(b)
        after=p.stat()
        if read!=st.st_size or (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('unstable')
        entries=index(q);rows=[r for r in labels if r['archive']=='VL'+name[2:]]
        if any(r['farm_id'] not in ALLOWED for r in rows):raise ValueError('reserved farm')
        lookup={norm(PurePosixPath(r['member']).stem):r for r in rows}
        if len(lookup)!=len(rows) or set(lookup)!={e['stem'] for e in entries}:raise ValueError('not bijective JSON-stem mapping')
        for e in entries:
            r=lookup[e['stem']]
            if e['stem'].rsplit('_',1)[-1]!=str(r['image_id']):raise ValueError('image id disagreement')
            mapped.append(dict(r,source_archive=name,source_member=e['member'],offset=e['offset'],bytes=e['bytes'],payload_sha256=e['sha256'],source_mapping='JSON_member_stem+image_id'))
        a={'name':name,'bytes':read,'source_mtime_ns':st.st_mtime_ns,'sha256':h.hexdigest(),'images':len(entries),'labels':len(rows),'metadata_fname_stem_matches':sum(norm(PurePosixPath(r['fname']).stem) in {e['stem'] for e in entries} for r in rows)}
        archives.append(a);print(json.dumps(a,ensure_ascii=False),flush=True)
    reference=[]
    previous=json.loads((RUN.parent/'image_ingestion_20261010_v1/source_qa_v1.json').read_text(encoding='utf-8'))
    for a in previous['archives']:
        p=OLD/a['archive']
        if hashfile(p)!=a['sha256']:raise ValueError('old snapshot changed')
        reference.extend(dict(e,archive=p.name) for e in index(p))
    hashes=collections.Counter(r['payload_sha256'] for r in mapped);refhash={r['sha256'] for r in reference}
    save('mapping_v1.json',{'archives':archives,'mapped':mapped,'new_requested_upper_bytes':requested,'cross_TS_exact_matches':[r['source_member'] for r in mapped if r['payload_sha256'] in refhash],'within_VS_duplicate_sha_groups':sum(n>1 for n in hashes.values()),'reference_TS_images':len(reference),'io_ledger':dict(IO),'scope':'two development VS and two cached TS only; no all-data uniqueness claim'})
    save('reference_index_v1.json',reference)
    save('selection_v1.json',{'rows':select_rows(mapped),'policy':'sorted farm/prefix round robin, chronological within group; first64; no image viewing used'})
def select_rows(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['farm_id'],r['prefix_candidate'])].append(r)
    for k in groups:groups[k].sort(key=lambda r:(r['date_captured'],r['image_id']))
    selected=[]
    for rank in range(max(map(len,groups.values()))):
        for k in sorted(groups):
            if rank<len(groups[k]):selected.append(groups[k][rank])
            if len(selected)==64:break
        if len(selected)==64:break
    return selected
def qa():
    from PIL import Image,ImageOps
    import io
    Image.MAX_IMAGE_PIXELS=25_000_000
    obj=json.loads((RUN/'mapping_v1.json').read_text(encoding='utf-8'));rows=obj['mapped']
    IO.update(obj['io_ledger'])
    for a in obj['archives']:
        if hashfile(LOCAL/a['name'])!=a['sha256']:raise ValueError('snapshot changed before QA')
    manifest=json.loads((RUN/'selection_v1.json').read_text(encoding='utf-8'))
    selected=manifest['rows']
    if selected!=select_rows(rows):raise ValueError('selection mismatch')
    if ownbytes()+128*1024**2>12*1024**3 or shutil.disk_usage(ROOT).free-128*1024**2<30*1024**3:raise ValueError('QA space gate')
    total_pixels=0
    results=[];output_bytes=0;out=LOCAL/'qa';out.mkdir(exist_ok=False)
    for r in selected:
        with (LOCAL/r['source_archive']).open('rb') as f:f.seek(r['offset']);b=tracked_read(f,r['bytes'])
        if hashlib.sha256(b).hexdigest()!=r['payload_sha256']:raise ValueError('selected payload changed')
        with Image.open(io.BytesIO(b)) as raw:
            size=list(raw.size);total_pixels+=raw.width*raw.height
            if raw.width*raw.height>25_000_000 or total_pixels>512_000_000:raise ValueError('pixel budget before decode')
            raw.load()
            if raw.format!='JPEG' or raw.width*raw.height>25_000_000:raise ValueError('decode budget/format')
            exif_orientation=raw.getexif().get(274);im=ImageOps.exif_transpose(raw).convert('RGB')
        gray=list(im.convert('L').resize((9,8),Image.Resampling.LANCZOS).getdata());dh=0
        for y in range(8):
            for x in range(8):dh=(dh<<1)|int(gray[y*9+x]>gray[y*9+x+1])
        im.thumbnail((1024,1024),Image.Resampling.LANCZOS);path=out/(r['source_archive'][:-4]+'__'+PurePosixPath(r['source_member']).stem+'.jpg');buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=90);encoded=buffer.getvalue()
        output_bytes+=len(encoded)
        if output_bytes>128*1024**2 or newbytes()+len(encoded)>2*1024**3 or shutil.disk_usage(ROOT).free-len(encoded)<30*1024**3:raise ValueError('QA output budget')
        with path.open('xb') as target:target.write(encoded)
        results.append(dict(r,actual_size=size,dimension_match=size==[int(r['width']),int(r['height'])],exif_orientation=exif_orientation,dhash=f'{dh:016x}',derived_path=str(path),derived_sha256=hashfile(path),derivation='QA thumbnail/JPEG90; source provenance label not validated'))
    near=[]
    for i,a in enumerate(results):
        for b in results[i+1:]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=6:near.append({'a':a['source_member'],'b':b['source_member'],'distance':distance,'same_farm':a['farm_id']==b['farm_id'],'same_candidate_prefix':a['prefix_candidate']==b['prefix_candidate']})
    collisions=json.loads((PREV/'filename_collision_v1.json').read_text(encoding='utf-8'))['groups'];byid={r['image_id']:r for r in rows};checks=[]
    for c in collisions:
        rr=[byid[r['image_id']] for r in c['rows'] if r['image_id'] in byid]
        if rr:checks.append({'fname':c['normalized_fname'],'mapped_rows':len(rr),'total_collision_rows':len(c['rows']),'ids':[r['image_id'] for r in rr],'payload_hashes':[r['payload_sha256'] for r in rr],'distinct_payloads':len({r['payload_sha256'] for r in rr}),'status':'preserved; same/different pixels require further comparison'})
    save('qa_v1.json',{'selected':results,'groups':len({(r['farm_id'],r['prefix_candidate']) for r in results}),'farm_counts':dict(collections.Counter(r['farm_id'] for r in results)),'io_ledger':dict(IO),'QA_output_bytes':output_bytes,'decoded_pixels':total_pixels,'dimension_mismatches':sum(not r['dimension_match'] for r in results),'near_candidates':near,'collision_mapping':checks,'scope':'64 development decodes; dHash candidates not proven duplicates; no model training'})
    print(json.dumps({'decoded':len(results),'groups':len({(r['farm_id'],r['prefix_candidate']) for r in rows}),'near_candidates':len(near),'dimension_mismatches':sum(not r['dimension_match'] for r in results)},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mapping','qa'],default='mapping',nargs='?');a=p.parse_args();mapping() if a.stage=='mapping' else qa()
