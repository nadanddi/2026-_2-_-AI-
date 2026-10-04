"""Prepare a new own-local overlay only. No compile, install, data read or fit."""
from pathlib import Path
import hashlib,json,shutil,difflib,subprocess,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_exact_tweedie_leaf_20261004_v1'
SOURCE=LOCAL/'source';DEST=LOCAL/'overlay_v1'
ref=json.loads((H/'source_reference_v1.json').read_text())
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==ref['commit']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in ref['sha256'].items():assert sha(SOURCE/name)==digest
assert not DEST.exists()
shutil.copytree(SOURCE,DEST,ignore=shutil.ignore_patterns('.git'))
changes={}
for name in ref['sha256']:
 before=(SOURCE/name).read_text(encoding='utf-8');after=before
 if name.endswith('regression_objective.hpp'):
  after=after.replace('#include <vector>','#include <vector>\n#include "farmai_leaf_audit.hpp"',1)
  marker='#undef PercentileFun';assert after.count(marker)==1
  after=after.replace(marker,(H/'exact_leaf_class_v1.hpp').read_text(encoding='utf-8')+'\n'+marker)
 elif name.endswith('objective_function.cpp'):
  needle='return new RegressionTweedieLoss(config);\n'
  assert after.count(needle)==2
  after=after.replace(needle,needle+'  } else if (type == std::string("tweedie_exact_leaf")) {\n    return new RegressionTweedieExactLeaf(config);\n')
  needle='return new RegressionTweedieLoss(strs);\n';assert after.count(needle)==1
  after=after.replace(needle,needle+'  } else if (type == std::string("tweedie_exact_leaf")) {\n    return new RegressionTweedieExactLeaf(strs);\n')
 elif name.endswith('config.cpp'):
  needle='ParseMetrics(objective, metric);';assert after.count(needle)==1
  after=after.replace(needle,'ParseMetrics(objective == "tweedie_exact_leaf" ? "tweedie" : objective, metric);')
 elif name.endswith('bagging.hpp'):
  after=after.replace('#include <vector>','#include <vector>\n#include "../objective/farmai_leaf_audit.hpp"',1)
  needle='// set bagging data to tree learner';assert after.count(needle)==1
  after=after.replace(needle,'FarmaiBagAudit(iter, bag_data_indices_.data(), bag_data_cnt_);\n      '+needle)
 assert after!=before
 (DEST/name).write_text(after,encoding='utf-8',newline='\n')
 changes[name]=dict(original_sha256=sha(SOURCE/name),overlay_sha256=sha(DEST/name))
 with (H/(Path(name).name+'.patch')).open('x',encoding='utf-8') as f:f.writelines(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
shutil.copyfile(H/'farmai_leaf_audit_v1.hpp',DEST/'src/objective/farmai_leaf_audit.hpp')
with (H/'overlay_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_OVERLAY_PREP_ONLY',commit=ref['commit'],changes=changes,extra_header_sha256=sha(DEST/'src/objective/farmai_leaf_audit.hpp'),patch_script_sha256=sha(Path(__file__)),class_sha256=sha(H/'exact_leaf_class_v1.hpp'),compile=0,fit=0),f,indent=2)
print('OVERLAY_PREP_PASS_FIT0')
