"""Own-local portable build only; no global install or model fitting."""
from pathlib import Path
import os,subprocess,json,hashlib,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_exact_tweedie_leaf_20261004_v1'
TOOL=LOCAL/'toolchain_v1/mingw64/bin';SOURCE=LOCAL/'overlay_v1';BUILD=LOCAL/'build_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(LOCAL/'winlibs.zip')=='d5dbafc4a170e762ca6143151ec918fb9e2c72736fb14cd704abebc6bdd5276a'
manifest=json.loads((H/'overlay_manifest_v1.json').read_text())
for name,item in manifest['changes'].items():assert sha(SOURCE/name)==item['overlay_sha256']
assert sha(SOURCE/'src/objective/farmai_leaf_audit.hpp')==manifest['extra_header_sha256']
assert not BUILD.exists();BUILD.mkdir()
env=os.environ.copy();env['PATH']=str(TOOL)+os.pathsep+env.get('PATH','')
cmake=str(TOOL/'cmake.exe')
commands=[[cmake,'-S',str(SOURCE),'-B',str(BUILD),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DUSE_GPU=OFF','-DUSE_CUDA=OFF','-DUSE_OPENMP=ON','-DBUILD_CLI=OFF','-D__BUILD_FOR_PYTHON=ON','-DCMAKE_C_COMPILER='+str(TOOL/'gcc.exe'),'-DCMAKE_CXX_COMPILER='+str(TOOL/'g++.exe'),'-DCMAKE_MAKE_PROGRAM='+str(TOOL/'ninja.exe')],[cmake,'--build',str(BUILD),'--parallel','2']]
result=dict(status='PENDING',commands=commands,compiler_sha256=sha(TOOL/'g++.exe'),cmake_sha256=sha(TOOL/'cmake.exe'),overlay_manifest_sha256=sha(H/'overlay_manifest_v1.json'),fit=0,install=0)
try:
 for i,cmd in enumerate(commands):
  with (H/f'build_stage_{i}_v1.log').open('x',encoding='utf-8') as log:
   completed=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,encoding='utf-8')
  assert completed.returncode==0,('build stage failed',i,completed.returncode)
 result['status']='PASS_BUILD_ONLY';dlls=list(LOCAL.glob('overlay_v1/lib_lightgbm.dll'))+list(BUILD.rglob('lib_lightgbm.dll'))
 assert len(dlls)==1,dlls
 result['dll']=str(dlls[0]);result['dll_sha256']=sha(dlls[0])
except BaseException as e:result['status']='FAIL_BUILD_ONLY';result['error']=str(e);raise
finally:
 with (H/'build_result_v1.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2)
print(json.dumps(result,indent=2))
