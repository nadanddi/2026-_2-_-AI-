"""Same pre-registered fold recipes, separate CPU processes; checkpoint resume."""
import sys,subprocess,concurrent.futures,json,time,ctypes
from pathlib import Path

def worker(name,index):
    import run_benchmark as b
    b.log=lambda msg:print(msg,flush=True)
    raw,full,lab,lock,sigs,fds=b.prepare()
    unit=[z for z in fds if z[0]==name and z[1]==index]
    assert len(unit)==1
    _,_,days=unit[0]
    va=lab[[(f,int(d)) in days for f,d in zip(lab.farm,lab.day)]].reset_index(drop=True)
    tr=lab[b.near_mask(lab,days|lock)].reset_index(drop=True)
    with b.threadpool_limits(limits=4):
        b.torch.set_num_threads(4)
        b.fit_fold(tr,va,name,index)

def coordinator():
    import run_benchmark as b
    raw,full,lab,lock,sigs,fds=b.prepare()
    # Keep enough RAM for each offline TabPFN fit and leave the OS headroom.
    class Mem(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('page_total',ctypes.c_ulonglong),('page_available',ctypes.c_ulonglong),('virtual_total',ctypes.c_ulonglong),('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
    mem=Mem();mem.length=ctypes.sizeof(mem);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    workers=max(1,min(3,int((mem.available/2**30-2)/2.5)))
    print(f'Workers={workers}, available RAM={mem.available/2**30:.1f}GB',flush=True)
    units=[]
    for name,i,_ in fds:
        p=b.OUT/f'{name}_{i}.npz'
        if p.exists() and p.with_suffix('.json').exists():
            m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert b.sha(p)==m['prediction_sha256']
        else:units.append((name,i))
    b.save(b.HERE/'parallel_execution.json',{'workers':workers,'available_ram_gb':mem.available/2**30,'units_remaining':units,
              'recipe_changes':False,'completed_fold_preserved':True})
    def launch(unit):
        name,i=unit
        with (b.OUT/f'worker_{name}_{i}.log').open('w',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,'-u',str(Path(__file__)),name,str(i)],stdout=f,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        if p.returncode:raise RuntimeError(f'worker {name}/{i} exit {p.returncode}; see worker log')
        print(f'COMPLETED {name}/{i}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(launch,u) for u in units]
        for f in concurrent.futures.as_completed(futures):f.result()
    # All folds are complete. Original runner validates caches and assembles OOF.
    b.main()

if __name__=='__main__':
    if len(sys.argv)==3:worker(sys.argv[1],int(sys.argv[2]))
    else:coordinator()
