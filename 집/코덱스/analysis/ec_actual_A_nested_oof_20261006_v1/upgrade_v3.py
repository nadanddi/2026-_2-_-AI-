from pathlib import Path
H=Path(__file__).resolve().parent
def write(name,text):
    with (H/name).open('x',encoding='utf-8') as f:f.write(text)
r=(H/'run_v2.py').read_text(encoding='utf-8')
for old,new in [('run_v2.py','run_v3.py'),('verify_v2.py','verify_v3.py'),('preparation_v2.json','preparation_v3.json'),('registration_v2.json','registration_v3.json'),('receipt_v2.json','receipt_v3.json')]:r=r.replace(old,new)
r=r.replace('import sys,os,json,hashlib','import uuid\nimport sys,os,json,hashlib')
old="""def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
"""
new="""def save(p,x):
    p=Path(p)
    if p.exists():assert load(p)==x;return
    stage=p.with_name(p.name+'.partial_'+uuid.uuid4().hex)
    with stage.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
    os.rename(stage,p)
def cp(name):return OUT/'components'/Path(name).stem/Path(name).name
def savecsv(p,df):
    if p.exists():pd.testing.assert_frame_equal(pd.read_csv(p,float_precision='round_trip'),df,check_dtype=False,check_exact=True);return
    stage=p.with_name(p.name+'.partial_'+uuid.uuid4().hex)
    with stage.open('x',encoding='utf-8',newline='') as f:df.to_csv(f,index=False)
    os.rename(stage,p)
"""
assert old in r;r=r.replace(old,new)
r=r.replace("def cache(path,signature,row_ids):\n", "def cache(path,signature,row_ids):\n    path=cp(path.name)\n")
old="""    with path.open('xb') as f:np.savez(f,**d)
    m=dict(signature=signature,sha=sha(path),details=details);save(path.with_suffix('.json'),m);return d,m
"""
new="""    dest=cp(path.name);dest.parent.parent.mkdir(parents=True,exist_ok=True)
    stage=OUT/'.staging'/(dest.parent.name+'_'+uuid.uuid4().hex);stage.mkdir(parents=True,exist_ok=False);temp=stage/dest.name
    with temp.open('xb') as f:np.savez(f,**d)
    m=dict(signature=signature,sha=sha(temp),details=details,path=str(dest.relative_to(OUT)))
    save(temp.with_suffix('.json'),m);assert sha(temp)==m['sha']
    os.rename(stage,dest.parent);return d,m
"""
assert old in r;r=r.replace(old,new)
r=r.replace("files.append(dict(path=f'{v}_{k}_{j}_r3_{s}.npz',sha=m['sha']))", "files.append(dict(path=m['path'],sha=m['sha']))")
r=r.replace("files.append(dict(path=f'{v}_{k}_{j}_pfn_{s}.npz',sha=m['sha']))", "files.append(dict(path=m['path'],sha=m['sha']))")
old="""                if dest.exists():old=pd.read_csv(dest,float_precision='round_trip');pd.testing.assert_frame_equal(old,df,check_dtype=False)
                else:df.to_csv(dest,index=False)
"""
assert old in r;r=r.replace(old,"                savecsv(dest,df)\n")
r=r.replace("dest=OUT/'nested_cases_v1.csv';assert not dest.exists();case.to_csv(dest,index=False)","dest=OUT/'nested_cases_v1.csv';savecsv(dest,case)")
r=r.replace("dest=OUT/'nested_seed_consensus_v1.csv';assert not dest.exists();pd.DataFrame(consensus).to_csv(dest,index=False)","dest=OUT/'nested_seed_consensus_v1.csv';savecsv(dest,pd.DataFrame(consensus))")
write('run_v3.py',r)
v=(H/'verify_v2.py').read_text(encoding='utf-8')
for old,new in [('run_v2.py','run_v3.py'),('verify_v2.py','verify_v3.py'),('preparation_v2.json','preparation_v3.json'),('registration_v2.json','registration_v3.json'),('receipt_v2.json','receipt_v3.json'),('verification_v2.json','verification_v3.json'),('학습산출물_검산_v2.md','학습산출물_검산_v3.md'),('reproduction_v2.zip','reproduction_v3.zip'),('bundle_v2.json','bundle_v3.json')]:v=v.replace(old,new)
v=v.replace('import sys,json,csv','import uuid,os\nimport sys,json,csv')
old="""def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
"""
new="""def save(p,x):
    p=Path(p)
    if p.exists():assert load(p)==x;return
    stage=p.with_name(p.name+'.partial_'+uuid.uuid4().hex)
    with stage.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
    os.rename(stage,p)
def cp(name):return OUT/'components'/Path(name).stem/Path(name).name
"""
assert old in v;v=v.replace(old,new)
v=v.replace("path=OUT/name", "path=cp(name)")
v=v.replace("arrays(OUT/f'{v}_{k}_{j}_r3_{s}.npz')", "arrays(cp(f'{v}_{k}_{j}_r3_{s}.npz'))").replace("arrays(OUT/f'{v}_{k}_{j}_pfn_{c}.npz')", "arrays(cp(f'{v}_{k}_{j}_pfn_{c}.npz'))")
v=v.replace("data=rows(OUT/f'OOF_{v}_{k}_{s}.csv');assert", """data=rows(OUT/f'OOF_{v}_{k}_{s}.csv')
            for rr in data:
                assert all(math.isfinite(float(rr[field])) for field in ['y','y_day','A','raw_r3','raw_pfn','raw_A','clip_lo','clip_hi','prefix_A']),rr['row_id']
            assert""")
old="""    with (H/'학습산출물_검산_v3.md').open('x',encoding='utf-8') as f:f.write('\\n'.join(description)+'\\n')
    zp=OUT/'reproduction_v3.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_v3.py','verify_v3.py','plan_v1.md','registration_v3.json','preparation_v3.json','receipt_v3.json','verification_v3.json','학습산출물_검산_v3.md']:z.write(H/name,'analysis/'+name)
        for f in receipt['files']:
            p=OUT/f['path'];z.write(p,'local/'+p.name)
            if p.suffix=='.npz':z.write(p.with_suffix('.json'),'local/'+p.with_suffix('.json').name)
"""
new="""    doc=H/'학습산출물_검산_v3.md';text='\\n'.join(description)+'\\n'
    if doc.exists():assert doc.read_text(encoding='utf-8')==text
    else:
        stage=doc.with_name(doc.name+'.partial_'+uuid.uuid4().hex)
        with stage.open('x',encoding='utf-8') as f:f.write(text)
        os.rename(stage,doc)
    zp=OUT/'reproduction_v3.zip';entries={}
    for name in ['run_v3.py','verify_v3.py','plan_v1.md','registration_v3.json','preparation_v3.json','receipt_v3.json','verification_v3.json','학습산출물_검산_v3.md']:entries['analysis/'+name]=H/name
    for f in receipt['files']:
        p=OUT/f['path'];entries['local/'+Path(f['path']).as_posix()]=p
        if p.suffix=='.npz':entries['local/'+Path(f['path']).with_suffix('.json').as_posix()]=p.with_suffix('.json')
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            assert set(z.namelist())==set(entries) and z.testzip() is None
            for name,path in entries.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha(path)
    else:
        stage=zp.with_name(zp.name+'.partial_'+uuid.uuid4().hex)
        with zipfile.ZipFile(stage,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,path in entries.items():z.write(path,name)
        with zipfile.ZipFile(stage) as z:assert set(z.namelist())==set(entries) and z.testzip() is None
        os.rename(stage,zp)
"""
assert old in v;v=v.replace(old,new)
write('verify_v3.py',v)
pre=(H/'prelaunch_v3.py').read_text(encoding='utf-8').replace('preparation_v2.json','preparation_v3.json').replace('prelaunch_v3.json','prelaunch_v4.json');write('prelaunch_v4.py',pre)
print('UPGRADE_V3_CREATED')
