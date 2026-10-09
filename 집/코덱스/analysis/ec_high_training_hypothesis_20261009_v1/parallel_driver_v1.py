from pathlib import Path
import sys,json,os,hashlib
H=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(H));import run_v4 as P
PARTS={'early':[1,2,3,4,5],'late':[6,7,8,9]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(part):
    assert part in PARTS
    assert (H/'critique_midpoint_v1.md').exists() and (H/'critique_parallel_driver_v1.md').exists()
    driver=json.loads((H/'parallel_driver_registration_v1.json').read_text(encoding='utf8'))
    assert driver=={'source_sha':sha(__file__),'producer_sha':sha(H/'run_v4.py'),'registration_sha':sha(H/'registration_v4.json'),'parts':PARTS,'fit_change':False,'score':False}
    jobs,reg=P.prepare();assert json.loads(json.dumps(reg,ensure_ascii=False,allow_nan=False))==json.loads((H/'registration_v4.json').read_text(encoding='utf8'))
    lock=H/('worker_'+part+'_v1.lock')
    with lock.open('x',encoding='utf8') as f:json.dump({'pid':os.getpid(),'part':part,'driver_sha':sha(__file__)},f)
    P.run(jobs,PARTS[part])
    assert json.loads(lock.read_text(encoding='utf8'))['pid']==os.getpid();lock.unlink()
    print('PART_COMPLETE',part,PARTS[part],flush=True)
if __name__=='__main__':main(sys.argv[1])