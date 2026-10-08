from pathlib import Path
import json,hashlib,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['DOMAIN24_original_preparation_independent_crosscheck_v1.json',
       'DOMAIN24_original_raw_fit_registration_v3.json','critique_DOMAIN24_original_raw_actual_registration_v1.md',
       'DOMAIN24_original_raw_metadata_snapshot_v1.json','audit_original_raw_metadata_v1.py',
       'run_original_pfn_cache_v1.py','run_original_pfn_cache_v2.py','original_pfn_cache_rules_v1.py',
       'DOMAIN24_original_pfn_rules_synthetic_audit_v1.json','DOMAIN24_original_pfn_runtime_plan_v2.json',
       'DOMAIN24_original_pfn_registration_v2.json','critique_ORIGINAL_PFN_cache_plan_v2.md',
       'critique_ORIGINAL_PFN_actual_registration_v1.md']
assert all((HERE/n).exists() for n in names)
prep=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2/complete.json'
complete=json.loads(prep.read_text(encoding='utf-8'))
assert complete['folds']==66 and complete['prefix_checks']==396
rawroot=HERE/'checkpoints/DOMAIN24_ORIGINAL_RAW_v3'
snap={str(p.relative_to(rawroot)):sha(p) for p in rawroot.glob('*_fold*/*_seed*.json')}
record={'checkpoint':20,'files_sha256':{n:sha(HERE/n) for n in names},
        'preparation66_original_handle77114_exit_code':0,'preparation_complete_sha256':sha(prep),
        'independent_preparation_actual_exit0':True,'raw_fit_registration_pins':214,
        'raw_cpu_worker_session':93925,'raw_worker_last_poll':'running; DIAG10/0 seed47 D11 PASS',
        'raw_completed_file_snapshot':snap,'partial_raw_metadata_verified_files':26,
        'PFN_fit_registration_pins':228,'PFN_context_fits_registered':264,'PFN_model_fits':0,
        'PFN_start_deferred':'Measured free physical memory2282819584bytes; avoid concurrent raw/PFN memory peak',
        'memory_snapshot':{'logical_processors':16,'physical_total_bytes':16452853760,'physical_available_bytes':2282819584,'load_percent':86},
        'heldout_original_scores_read':False,'adoption':False,'GPU':False,'submission':False,'whole_goal_complete':False,
        'next':['Poll original93925; no duplicate restart from file/lock alone',
                'Recheck resource availability; run registered CPU PFN2 when sufficient headroom or raw worker completes',
                'All raw5346 and PFN264 context receipts require fresh full audit',
                'Full mixing/shrink/clip/SG2 causal gates BEFORE originalheldouttruth score',
                'All24 TM/P2LOO/EL1 seeds, CH2 original, literature24/data142/subsets/interactions, firstunused confirmation/report remain']}
with (HERE/'checkpoint_record_v20.json').open('x',encoding='utf-8') as h:json.dump(record,h,ensure_ascii=False,indent=2)
note=f'''
# 현재 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v20)

- 원prep77114 직접terminalexit0(마지막P2fold55);complete66/396생성. stdlib독립prep actualexit0/1584candidate서명/66ID·SHA/flagsPASS. registrar5 정식승인actualexit0→rawregistration_v3.json214pins. 독립actual등록214전부MATCH/coreblocker0.
- 원CPU run_domain_original_raw_v3.py 실제session93925 running. 마지막직접poll DIAG10fold0 seed47 baseline3+D01..D11 rawPASS. 파일스냅샷{len(snap)}/5346은부분이며현재live는원handle근거. 별도metadata stdlib읽기actualexit0/26완료파일 ID/등록·model/runtime/행렬署名/열·imputer/5diff계약PASS(독립freshmatrix/refit 아님). 채점0.
- 원PFN初안v1비평OPFN01/02→새v2+purecache rules:4cache/6stats/12KV/원CPUdevice/정확120trace·21checks/engine·실architecture1/cache4/memoryauto/actualimports. 합성21 actualPASS·CPU실runtime/context264 capture2 actualexit0. registrarPFN2 actualexit0/228pins/원raw214전이·264context2000rows 봉인,독립actual등록228MATCH/coreblocker0. PFNfit0.
- CPU16논리코어지만메모리実snapshot총16452853760/가용2282819584byte/load86%. raw+PFN동시최대메모리회피로PFN시작대기. 현재rawworker를중단/재시작하지않음. 메모리여유재확인 또는raw종료뒤 등록된 **run_original_pfn_cache_v2.py** 실행가능;GPU0.
- 다음원93925→5346 raw완료 확인;PFN2실행/264fit 및528파일후strict전체재검산·fullmix/shrink/clip/SG2/pretruthgate·원TM111/P2LOO/EL1×3seed채점. 文헌24/자료142/원subset16383·interaction/CH2원검증/최초미사용확정1회·전체보고서미완료. goal active/채택·제출0.

이하 이전 시점 기록.

'''.replace('署名','서명').replace('初안','초안').replace('実snapshot','실측').replace('文헌','문헌')
progress=HERE/'PROGRESS.md';progress.write_text(note+progress.read_text(encoding='utf-8'),encoding='utf-8')
for p in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
    with p.open('a',encoding='utf-8') as h:h.write(note)
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
assert max(int(x) for x in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))==383
entry='''
| 6.384 | **원66 특징준비 실제완료·원domain24 CPU학습 시작·PFN학습전용264문맥 등록** (2026-10-07 집 코덱스) | prep77114 terminalexit0/66·396,독립prep actual1584서명/ID·SHA PASS. registrar5실214핀/actualcritic모두MATCH뒤rawrunner3실93925 running,DIAG0 seed47baseline3+D11까지PASSlog;partialmetadata26파일계약 actualPASS/성능채점0. PFNv1재개구조비평→새v2 purecache4/6stats/12KV·CPUdevice/120trace/21checks/실architecture1/cache4,합성21PASS;실runtime/context264 capture2·등록PFN2 actual228핀/독립전부MATCH. PFNfit0/자원16core·실가용2.28GB라同時raw+PFN대기,메모리확보 또는raw종료뒤정식CPU실행;GPU0. raw/PFN완료만으로fullmix/SG2/원score허용안함;원全24×3검증/文헌·자료/subset·interaction/CH2/최초미사용확정/전체보고서미완료·goal active/채택·제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/DOMAIN24_original_raw_fit_registration_v3.json;DOMAIN24_original_raw_metadata_snapshot_v1.json;DOMAIN24_original_pfn_registration_v2.json;critique_ORIGINAL_PFN_actual_registration_v1.md;checkpoint_record_v20.json |
'''.replace('同時','동시').replace('全24','전체24').replace('文헌','문헌')
with catalog.open('a',encoding='utf-8') as h:h.write(entry)
print(f'checkpoint20/catalog6.384 saved; rawfiles snapshot{len(snap)}, CPUraw running, PFNfit0, goal incomplete')
