from pathlib import Path
import ast
here=Path(__file__).resolve().parent
source=(here/'original_fold_features_v1.py').read_text(encoding='utf-8')
source=source.replace('from blk_baseline_data_v1 import *','from blk_baseline_data_v1 import *\nimport blk_baseline_data_v1 as base_module\nimport blk_context_v1 as context_module\nimport domain_features_v2 as domain_module\nimport checkpoint_v1 as checkpoint_module')
marker="    regpath=HERE/'DOMAIN24_original_preparation_registration_v2.json'"
source=source.replace(marker,"""    for module,name in [(base_module,'blk_baseline_data_v1.py'),(context_module,'blk_context_v1.py'),
                        (domain_module,'domain_features_v2.py'),(checkpoint_module,'checkpoint_v1.py')]:
        assert Path(module.__file__).resolve()==(HERE/name).resolve()
"""+marker)
marker="if __name__=='__main__':"
source=source.replace(marker,"""def load_production_fold(validator,fold_number,fit_registration):
    # This API never exposes the probe flag. Every fit must require the sealed 66-fold completion.
    own=str(Path(__file__).resolve())
    assert fit_registration['status']=='REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT'
    assert fit_registration['source_sha256'][own]==sha(__file__)
    assert all(sha(p)==v for p,v in fit_registration['source_sha256'].items())
    folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
    assert sha(folder/'complete.json')==fit_registration['preparation_complete_sha256']
    bundle=load_fold(validator,fold_number,require_all66=True)
    assert bundle[-1]['require_all66'] is True
    assert bundle[-1]['loader_source_sha256']==fit_registration['source_sha256'][own]
    return bundle

"""+marker)
source=source.replace("assert sys.argv[1:]==['--probe','DIAG10','0']", "assert sys.argv[1:]==['--probe','DIAG10','0']")
source=source.replace('DOMAIN24_original_matrix_reconstruction_probe_v1.json','DOMAIN24_original_matrix_reconstruction_probe_v2.json')
ast.parse(source);out=here/'original_fold_features_v2.py';assert not out.exists();out.write_text(source,encoding='utf-8')
print('Materializer2: actual module paths and mandatory sealed production66 API added; no fit')
