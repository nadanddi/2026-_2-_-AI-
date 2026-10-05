"""One-off completion helper for the already-running, immutable run_v4 worker."""
from pathlib import Path
import json,sys,hashlib,ctypes,subprocess,os,time
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    reg=load(H/'verification_registration_v5.json')
    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name
    launch=load(H/'launch_v4.json');pid=launch['pid'];assert sha(H/'run_v4.py')==launch['source_sha']
    lock=OUT/'worker.lock';assert load(lock)['pid']==pid
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle.restype=ctypes.c_int
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    print('WAITING_EXISTING_WORKER',pid,'NO_RESTART',flush=True)
    try:
        while True:
            state=kernel.WaitForSingleObject(handle,30000)
            if state==0:break
            if state!=258:raise RuntimeError(('WaitForSingleObject',state))
    finally:kernel.CloseHandle(handle)
    receipt=load(H/'receipt_v4.json');assert receipt['status']=='COMPLETE_80_CONTEXTS_ACTUAL_A_OOF'
    subprocess.run([sys.executable,str(H/'verify_v5.py')],check=True)
    proof=load(H/'verification_v5.json');assert proof['status']=='PASS_COMPLETE_ACTUAL_A_NESTED_OOF'
    bundle=load(H/'bundle_v5.json');assert sha(OUT/bundle['path'])==bundle['sha']
    if lock.exists():assert load(lock)['pid']==pid;lock.unlink()
    save(H/'finalization_v5.json',dict(status='COMPLETE_TRAINING_AND_NUMERIC_VERIFICATION_CRITIC_REVIEW_PENDING',worker_pid=pid,receipt_sha=sha(H/'receipt_v4.json'),proof_sha=sha(H/'verification_v5.json'),bundle_sha=bundle['sha'],source_sha=sha(Path(__file__)),classifier_fit=False,submission=False,limitation='Independent final adversarial critique still required before reporting conclusions'))
    print('COMPLETE_NUMERIC_VERIFICATION_FINAL_CRITIQUE_PENDING',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        save(H/'finalization_failure_v5.json',dict(status='STOP_PRESERVE_EXISTING_WORKER_ARTIFACTS',error_type=type(error).__name__,error=str(error)));raise
