# -*- coding: utf-8 -*-
"""CLN7 (2026-10-16 집 클로드; 사용자 'p .079 는 좀 그렇다, 최소 유의 수준은 맞춰야' -> 5점 정리판 핵심(BASE 어제 재연결)을 고를 때 안 쓴 새 배치로 확인 시험).
사전 고정(커밋 후 실행).  온도.  BASE 학습 산출(판정은 CLN7J, 사전 고정).
배치: N·CK2(OS1 v2 layout_folds / masks), CV8(cv8_cold_validator layout_folds / masks), EL1(Codex TK2 world sets, common.split_mask) — CLN1~6 판정(EXT10·EXT12·DIAG10)에 안 쓴 배치.
평가형 세계(RS1 장치: 판정 기록 입력을 학습 특징에서 지움)에서 같은 폴드마다 두 BASE:
  NUM = 번호 순(링커 없음, 원래 세계) / RS = FL 링커(p >= .5; 학습·평가 기록 fl3 표, 판정 가림 기록의 첫 연결은 그 배치·폴드의 FL7 거울 표)로 같은 동 어제 재연결, 연결 없으면 멈춤.
학습 함수·특징·가중·시드(7, 101) = CLN2·CLN4N 과 같음.  -> local/cln7_ckpt/{배치}_{폴드}.pkl (row_id, base_NUM_7·101, base_RS_7·101)
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln7_confirm_restitch_base_v1.py list | N:0,1 CK2:0 ...
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ARGS = sys.argv[1:]; sys.argv = ["x"]
def _load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
C2 = _load("cln2", "cln2_fl_restitch_base_v1.py"); R1, EVP, TF1, W, common = C2.R1, C2.EVP, C2.TF1, C2.W, C2.common
O1 = _load("os1", "os1_temp_regime_split_v2.py"); C8 = _load("cv8", "cv8_cold_validator_v1.py")
CK = os.path.join(env.LOCAL, "cln7_ckpt"); P_MIN = .5; SEEDS = (7, 101)
def key(r): return (r[:3], int(r[4:7]))
class RawNum(R1.Raw):
    def __init__(self): super().__init__(); self.link = {}
class RawFL7(C2.RawFL):
    def set_fold(self, L, k):
        M7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); M7 = M7[(M7.layout == L) & (M7.fold == k)]
        self.link = dict(self.base_link); self.mirror = {key(r): (key(p) if isinstance(p, str) and q >= P_MIN else None) for r, p, q in zip(M7.rec, M7.prev, M7.p)}
def folds_and_masks(lab, sets):
    LF = O1.layout_folds(W, lab); F8 = C8.layout_folds(W, lab)["CV8"]
    return {"N": (LF["N"], O1.masks), "CK2": (LF["CK2"], O1.masks), "CV8": (F8, C8.masks), "EL1": (dict(sets)["EL1"], common.split_mask)}
def build(raw, lab, tm, vm, ct, phc, wmap):
    labrec = set(zip(lab.farm, lab.day.astype(int))); val = set(zip(lab.farm[vm], lab.day[vm].astype(int))); trn = set(zip(lab.farm[tm], lab.day[tm].astype(int))); dele = labrec - val - trn
    vis_tr = lambda r: (lambda x: x in raw.train and x not in val and x not in dele)
    def vis_va(r):
        def v(x):
            if x in dele: return False
            if x in val and (x[0] != r[0] or x[1] >= r[1]): return False
            if x in raw.test and x[0] != r[0]: return False
            return True
        return v
    st = []; TR = R1.build_rows(raw, sorted(trn), val | dele, True, vis_tr, ct, phc, []); VA = R1.build_rows(raw, sorted(val), dele, False, vis_va, ct, phc, st)
    w = TR.row_id.map(wmap).values.astype(float); assert np.isfinite(w).all()
    return VA, {s: TF1.fit_base(TR, ct, phc, w, s)(VA) for s in SEEDS}, st
def main():
    os.makedirs(CK, exist_ok=True)
    C = json.load(open(os.path.join(EVP.AUD, "cols.json"))); ct, phc = C["ct"], C["phc"]
    L0 = pd.read_pickle(os.path.join(EVP.AUD, "lab_feats.pkl")); wmap = dict(zip(L0.row_id, L0.w))
    lab, pfn, _, _, wb, wp, wv, sets, z = W.worlds(); FM = folds_and_masks(lab, sets)
    if ARGS and ARGS[0] == "list":
        for L, (folds, mk) in FM.items(): print(L, "folds", len(folds), "val rows", [int(mk(lab, fd)[1].sum()) for fd in folds])
        return
    rn = RawNum(); rf = RawFL7()
    for a in ARGS:
        L, ks = a.split(":"); folds, mk = FM[L]
        for k in map(int, ks.split(",")):
            path = os.path.join(CK, "%s_%d.pkl" % (L, k))
            if os.path.exists(path): continue
            t0 = time.time(); tm, vm = mk(lab, folds[k]); rn.link = {}; rf.set_fold(L, k)
            VA, pn, _ = build(rn, lab, tm, vm, ct, phc, wmap); VA2, pr, st = build(rf, lab, tm, vm, ct, phc, wmap)
            out = VA[["row_id"]].copy()
            for s in SEEDS: out["base_NUM_%d" % s] = pn[s]
            o2 = VA2[["row_id"]].copy()
            for s in SEEDS: o2["base_RS_%d" % s] = pr[s]
            out = out.merge(o2, on="row_id", how="inner"); out.to_pickle(path)
            pd.DataFrame(st).assign(layout=L, fold=k).to_csv(os.path.join(CK, "chains_%s_%d.csv" % (L, k)), index=False)
            print("%s/%d done | VA rows %d | RS depth mean %.2f depth0 %d | %.0fs" % (L, k, len(out), np.mean([s["depth"] for s in st]), sum(s["depth"] == 0 for s in st), time.time() - t0), flush=True)
if __name__ == "__main__":
    main()
