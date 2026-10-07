"""Exclusive writer and preservation of partial cells on top of receipt v1."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import tempfile
import time
from checkpoint_v1 import REQUIRED,digest,complete as v1_complete,reuse as v1_reuse

def start_ticks(pid):
    if os.name!='nt':raise RuntimeError('process creation check currently Windows only')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:raise ProcessLookupError('PID not accessible; do not assume stale')
    times=[wintypes.FILETIME() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):raise OSError('creation time unavailable')
        return (times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
    finally:kernel.CloseHandle(handle)

def validate(contract):
    if set(contract)!=REQUIRED:raise ValueError('schema')
    for k,v in contract.items():
        if k.endswith('sha256'):
            if not isinstance(v,str) or len(v)!=64:raise ValueError('hash length')
            try:bytes.fromhex(v)
            except ValueError:raise ValueError('hash hex')
    if not isinstance(contract['seed'],int) or not isinstance(contract['fold'],int):raise ValueError('seed/fold')

def complete(folder,contract,ids,y,pred):
    validate(contract);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'created_at':time.time(),'contract_sha256':digest(contract)}
    try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:raise RuntimeError('writer lock exists: preserve and audit PID/start/contract; no automatic overwrite')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(token,f);f.flush();os.fsync(f.fileno())
        if (folder/'receipt.json').exists():return v1_reuse(folder,contract)
        if (folder/'predictions.json').exists():raise RuntimeError('partial predictions preserved, cannot overwrite')
        return v1_complete(folder,contract,ids,y,pred)
    finally:
        # Remove only this exact lock file, never a competing or stale owner's lock.
        if json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='exclusive_checkpoint_synthetic_',dir=here) as tmp:
        folder=Path(tmp).resolve();assert folder.is_relative_to(here.resolve())
        ids=['synthetic_a'];c={k:digest(k) for k in REQUIRED}
        c.update(seed=47,fold=0,validator='SYNTHETIC',candidate_id='SYNTHETIC',query_ids_sha256=digest(ids))
        locked=folder/'locked';locked.mkdir()
        lock=locked/'WRITER_LOCK.json';lock.write_text(json.dumps({'pid':os.getpid(),'start_ticks':start_ticks(os.getpid())}))
        try:complete(locked,c,ids,[0],[.1])
        except RuntimeError:pass
        else:raise AssertionError('duplicate writer accepted')
        partial=folder/'partial';partial.mkdir();(partial/'predictions.json').write_text('preserve this partial')
        try:complete(partial,c,ids,[0],[.1])
        except RuntimeError:pass
        else:raise AssertionError('partial overwritten')
        assert (partial/'predictions.json').read_text()=='preserve this partial'
        bad=dict(c);bad['code_sha256']='z'*64
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError('nonhex accepted')
        done=folder/'done';r=complete(done,c,ids,[0],[.1]);assert complete(done,c,ids,[0],[.1])==r
    out=here/'checkpoint_selftest_v2.json';assert not out.exists()
    out.write_text(json.dumps({'status':'PASS','checks':['exclusive writer refusal','partial preserved','nonhex refusal','process start ticks observed','completed same-contract reuse'],
        'limitations':['동시writer 충돌 분기 합성 검사이며 두 실제 process 동시쓰기 검사는 미실행',
            '실제 학습 crash/stale PID 강제중단·복구는 미실행. stale lock 자동 삭제 안 함']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('exclusive checkpoint synthetic checks PASS')
