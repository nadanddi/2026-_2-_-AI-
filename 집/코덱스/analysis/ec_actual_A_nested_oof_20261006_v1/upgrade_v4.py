from pathlib import Path
H=Path(__file__).resolve().parent
def write(n,t):
    with (H/n).open('x',encoding='utf-8') as f:f.write(t)
r=(H/'run_v3.py').read_text(encoding='utf-8')
for old,new in [('verify_v3.py','verify_v4.py'),('preparation_v3.json','preparation_v4.json'),('registration_v3.json','registration_v4.json'),('receipt_v3.json','receipt_v4.json')]:r=r.replace(old,new)
write('run_v4.py',r)
v=(H/'verify_v3.py').read_text(encoding='utf-8')
for old,new in [('run_v3.py','run_v4.py'),('verify_v3.py','verify_v4.py'),('preparation_v3.json','preparation_v4.json'),('registration_v3.json','registration_v4.json'),('receipt_v3.json','receipt_v4.json'),('verification_v3.json','verification_v4.json'),('学習','학습'),('학습산출물_검산_v3.md','학습산출물_검산_v4.md'),('reproduction_v3.zip','reproduction_v4.zip'),('bundle_v3.json','bundle_v4.json')]:v=v.replace(old,new)
needle="    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name\n"
extra="""    import platform,pandas,sklearn,lightgbm,torch
    from tabpfn import TabPFNRegressor
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,torch=str(torch.__version__))
    assert versions==prep['runtime']['versions']
    for name,mod in [('numpy',np),('pandas',pandas),('sklearn',sklearn),('lightgbm',lightgbm),('torch',torch)]:assert sha(mod.__file__)==prep['runtime']['module_init_sha'][name]
    assert sha(sys.modules[TabPFNRegressor.__module__].__file__)==prep['runtime']['tabpfn_sha']
"""
assert needle in v;v=v.replace(needle,needle+extra)
needle="            for rr in data:\n"
extra="""                rid=rr['row_id'];assert rr['farm']==rid[:3] and int(rr['day'])==int(rid[4:7]) and int(rr['hour'])==int(rid[8:10])
                assert rr['v']==v and int(rr['k'])==k and int(rr['s'])==s
"""
assert needle in v;v=v.replace(needle,needle+extra)
v=v.replace("gaps=dict(A=0.,prefix_A=0.,raw_A=0.,PFN=0.)", "gaps=dict(A=0.,prefix_A=0.,raw_A=0.,PFN=0.,raw_R3=0.)")
needle="                    gaps['PFN']=max"
extra="                    gaps['raw_R3']=max(gaps['raw_R3'],abs(float(r['raw_r3'])-float(z['raw'][i])))\n"
assert needle in v;v=v.replace(needle,extra+needle)
needle="        assert proof==load(H/'verification_v4.json')\n"
extra="""        bundle=load(H/'bundle_v4.json');zp=OUT/bundle['path'];assert sha(zp)==bundle['sha']
        with zipfile.ZipFile(zp) as z:assert z.testzip() is None
"""
assert needle in v;v=v.replace(needle,needle+extra)
write('verify_v4.py',v)
pre=(H/'prelaunch_v4.py').read_text(encoding='utf-8').replace("H/'preparation_v3.json'","H/'preparation_v4.json'").replace("ec_hardcase_crossfit_20261005_v1/preparation_v3.json","ec_hardcase_crossfit_20261005_v1/preparation_v1.json").replace('prelaunch_v4.json','prelaunch_v5.json');write('prelaunch_v5.py',pre)
print('V4_CREATED_MODEL_AND_SPLIT_UNCHANGED')
