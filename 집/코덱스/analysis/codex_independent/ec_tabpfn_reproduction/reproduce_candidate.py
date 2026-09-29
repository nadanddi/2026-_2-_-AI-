# -*- coding: utf-8 -*-
"""Offline reconstruction of temp round 5 + EC round 3/TabPFN v2 candidate."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import env  # noqa: F401  bundled portable env must be first project import

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
MODEL_DIR = PACKAGE / 'model'
MODEL = MODEL_DIR / 'tabpfn-v2-regressor.ckpt'
EXPECTED_MODEL_SHA256 = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'

# Optional preinstalled local dependency set for offline verification.  A normal
# package installation uses requirements.txt and leaves this unset.
extra_root = os.environ.get('AGRI_PACKAGE_DEPS_ROOT')
_dll_handles = []
if extra_root:
    root = Path(extra_root).resolve()
    libs = root / 'python'
    sys.path.insert(0, str(libs))
    if hasattr(os, 'add_dll_directory'):
        for path in list(libs.glob('*.libs')) + list(libs.glob('*/.libs')) + [root / 'msvc', root / 'extra/torch/lib']:
            if path.is_dir():
                _dll_handles.append(os.add_dll_directory(str(path)))
    sys.path.append(str(root / 'extra'))

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_MODEL_CACHE_DIR'] = str(MODEL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402
import tabpfn  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402
from tabpfn.model_loading import get_cache_dir  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import common  # noqa: E402
from common import USABLE, OUT_COLS  # noqa: E402
from harness import load, views  # noqa: E402
import feat_temp74 as T74  # noqa: E402
import feat_new  # noqa: E402
import features_v4 as F4  # noqa: E402
import train_flags_v6 as TF  # noqa: E402
from make_submission_v3 import (SEEDS, T_HUB, ET1, LGBP, SHRINK_L, m_lgbh, m_ridge,
                                m_nys, m_et, m_lgbtw, m_mlp, causal_shrink)  # noqa: E402
from make_submission_v4 import W_EC  # noqa: E402
from make_submission_v7 import fit_mean_w, W_TEMP, W_FLAG, W_NOISY  # noqa: E402

NAME = 'candidate_temp06_ec_tabpfn_v2_cpu.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert MODEL.is_file() and sha(MODEL) == EXPECTED_MODEL_SHA256
    assert get_cache_dir().resolve() == MODEL_DIR.resolve(), 'TabPFN cache did not point into package'
    output = PACKAGE / 'output'
    output.mkdir(exist_ok=True)
    destination = output / NAME
    manifest_path = output / 'manifest_tabpfn_v2_cpu.json'
    assert not destination.exists() and not manifest_path.exists(), 'refusing to overwrite output'

    panel, lab_t0, lab_e0 = load()
    tX, ty, sX = common.load_raw()
    view = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    segc, phc, fpc = F4.names(sg), F4.names(ph), F4.names(fp)
    f93 = T74.base74(view['temp']) + list(blocks['dew']) + list(blocks['event'])
    ct = f93 + segc + fpc
    f14 = [c for c in (list(USABLE) + ['day', 'hr_sin', 'hr_cos', 'midnight']) if c not in OUT_COLS]
    ce_et = f14 + fpc
    assert len(f14) == 14 and len(ce_et) == 38

    def attach(df):
        return (df.merge(ex, on='row_id', how='left').merge(sg, on='row_id', how='left')
                  .merge(ph, on='row_id', how='left').merge(fp, on='row_id', how='left'))

    lab_t = attach(lab_t0)
    lab_e = lab_e0.merge(fp, on='row_id', how='left')
    test = attach(panel[panel.is_test].reset_index(drop=True))
    test = test.set_index('row_id').loc[sX.row_id].reset_index()
    assert len(lab_e) == 9600 and len(test) == 1440
    weight = TF.row_weights(lab_t, W_FLAG, w_noisy=W_NOISY)

    ytemp = lab_t.sub_temp.values
    imp = SimpleImputer(strategy='median').fit(lab_t[phc])
    phys = LinearRegression().fit(imp.transform(lab_t[phc]), ytemp, sample_weight=weight)
    b_tr = phys.predict(imp.transform(lab_t[phc]))
    b_te = phys.predict(imp.transform(test[phc]))
    pt = (W_TEMP['resid_lgb'] * (b_te + fit_mean_w(m_lgbh, lab_t[ct], ytemp - b_tr, test[ct], weight))
          + W_TEMP['ridge'] * fit_mean_w(m_ridge, lab_t[ct], ytemp, test[ct], weight, 'ridge')
          + W_TEMP['nys'] * fit_mean_w(m_nys, lab_t[ct], ytemp, test[ct], weight, 'ridge'))
    print('TEMP fitted', flush=True)

    yec = lab_e.sub_ec.values
    ec_base = (W_EC['et'] * fit_mean_w(m_et, lab_e[ce_et], yec, test[ce_et])
               + W_EC['lgbtw'] * fit_mean_w(m_lgbtw, lab_e[f14], yec, test[f14])
               + W_EC['mlp'] * fit_mean_w(m_mlp, lab_e[f14], yec, test[f14]))
    low, high = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    ec_base_final = np.clip(causal_shrink(ec_base, test, SHRINK_L), low, high)
    print('EC baseline fitted', flush=True)

    Xtr = lab_e[ce_et].to_numpy(dtype=np.float32)
    Xte = test[ce_et].to_numpy(dtype=np.float32)
    members = []
    for seed in (1, 2, 3, 4):
        idx = np.random.default_rng(seed).choice(len(lab_e), size=2000, replace=False)
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device='cpu', n_estimators=4, random_state=seed,
            ignore_pretraining_limits=True, inference_precision=torch.float32)
        model.fit(Xtr[idx], yec[idx])
        member = np.asarray(model.predict(Xte), dtype=float)
        assert len(member) == len(sX) and np.isfinite(member).all()
        members.append(member)
        print(f'TabPFN context {seed} predicted', flush=True)
    bag = np.mean(members, axis=0)
    pec = np.clip(causal_shrink(.8 * ec_base + .2 * bag, test, .5), low, high)

    # References validate the implementation, never act as prediction inputs.
    r5 = pd.read_csv(PACKAGE / 'reference/submission_06.csv')
    r3 = pd.read_csv(PACKAGE / 'reference/submission_04.csv')
    assert r5.row_id.tolist() == r3.row_id.tolist() == sX.row_id.tolist()
    assert np.array_equal(np.round(pt, 6), r5.sub_temp.to_numpy(float)), 'temperature reference mismatch'
    assert np.array_equal(np.round(ec_base_final, 6), r3.sub_ec.to_numpy(float)), 'EC baseline mismatch'
    result = pd.DataFrame({'row_id': sX.row_id.values, 'sub_temp': pt, 'sub_ec': pec})
    assert result.columns.tolist() == ['row_id', 'sub_temp', 'sub_ec']
    assert np.isfinite(result[['sub_temp', 'sub_ec']].to_numpy(float)).all()
    result.to_csv(destination, index=False, encoding='utf-8', float_format='%.6f', lineterminator='\n')
    reread = pd.read_csv(destination)
    assert reread.row_id.tolist() == sX.row_id.tolist()
    assert np.isfinite(reread[['sub_temp', 'sub_ec']].to_numpy(float)).all()

    expected = pd.read_csv(PACKAGE / 'reference/candidate_temp06_ec_tabpfn_v2_cpu.csv')
    assert expected.row_id.tolist() == reread.row_id.tolist()
    assert np.array_equal(expected[['sub_temp', 'sub_ec']].to_numpy(float),
                          reread[['sub_temp', 'sub_ec']].to_numpy(float)), 'candidate reproduction mismatch'
    manifest = dict(name=NAME, python=platform.python_version(),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                  lightgbm=lgb.__version__, torch=torch.__version__, tabpfn=tabpfn.__version__),
                    inputs={n: sha(Path(common.DATA) / n) for n in ('train_X.csv', 'train_y.csv', 'test_X.csv')},
                    model_sha256=sha(MODEL), license_sha256=sha(MODEL_DIR / 'LICENSE.txt'),
                    output_sha256=sha(destination), baseline_seeds=list(SEEDS), context_seeds=[1, 2, 3, 4],
                    context_size=2000, n_estimators=4, device='cpu', inference_precision='float32',
                    ec_blend=[.8, .2], shrink=.5, clip=[low, high],
                    reference_temperature='PASS', reference_baseline_ec='PASS',
                    reference_candidate='PASS', external_api=False, hidden_labels_read=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'REPRODUCED {destination} SHA-256 {manifest["output_sha256"]}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
