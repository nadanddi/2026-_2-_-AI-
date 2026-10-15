# -*- coding: utf-8 -*-
"""CLN2 (2026-10-16 집 클로드; 5점 정리판 2단계 — CLN1K(7.031): 날 넘는 특징을 지우면 약 1.7% 손해 -> 지우지 않고 '같은 동 어제'로 다시 이어 계산).
사전 고정(커밋 후 실행).  온도.  BASE 학습 산출만(판정은 CLN2J).
RS1(6.625, rs1_restitch_v1.py)의 세계 다시 쓰기 장치를 그대로 쓰되 세 가지를 바꿈:
  (1) 링커 = 현장형 FL(번호·달력 특징 없음): 정답 기록 fl3_train_links + 평가 기록 fl3_eval_links, 확신 p >= .5(CH5 사슬과 같은 손익분기 문턱);
      판정 폴드 가림 기록 자신의 첫 연결은 폴드 거울 표 FL5(fl5_mirror_links, 그 배치·폴드) 로 덮어씀.
  (2) 연결이 없거나 p < .5 이면 '앞 번호'로 대신하지 않고 거기서 멈춤(이후 창 = 결측) — 번호 순서 의존 제거.
  (3) 검증 틀 = TV3 배치(DIAG10·EXT10·EXT12; Codex TK2 world 폴드, common.split_mask) — val = 그 폴드 가림 기록, del = 학습 마스크 밖 정답 기록.
  가시성·창(7일)·학습 함수(TF1.fit_base, 시드 7·101)·행 가중(lab_feats.pkl)·특징 목록(cols.json ct·phc) = RS1 과 같음.
-> local/cln2_ckpt/{배치}_{폴드}.pkl (row_id, base_RS_7, base_RS_101) + 연결 통계 csv.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln2_fl_restitch_base_v1.py DIAG10:0,1,2   (배치:폴드 목록; 여러 프로세스로 나눠 실행)
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ARGS = sys.argv[1:]; sys.argv = ["x"]
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R1 = _load("rs1", os.path.join(HERE, "rs1_restitch", "rs1_restitch_v1.py")); EVP, TF1 = R1.EVP, R1.TF1
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"); sys.path.insert(0, TKDIR)
import world as W  # noqa: E402
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "cln2_ckpt"); P_MIN = .5; SEEDS = (7, 101)
def key(r): return (r[:3], int(r[4:7]))
class RawFL(R1.Raw):
    def __init__(self):
        super().__init__()
        A = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))[["rec", "prev", "p"]], pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv"))[["rec", "prev", "p"]]])
        A = A[A.prev.notna() & (A.p >= P_MIN)]; self.base_link = {key(r): key(p) for r, p in zip(A.rec, A.prev)}; self.link = dict(self.base_link)
    def set_fold(self, L, k):
        M5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); M5 = M5[(M5.layout == L) & (M5.fold == k)]
        self.link = dict(self.base_link); self.mirror = {}
        for r, p, q in zip(M5.rec, M5.prev, M5.p):
            self.mirror[key(r)] = key(p) if isinstance(p, str) and q >= P_MIN else None
    def step(self, x):
        if x in getattr(self, "mirror", {}): return self.mirror[x]
        return self.link.get(x)                       # no link -> None (stop; never fall back to the previous number)
    def chain(self, r, visible):
        out, cur, seen = [], r, {r}
        for _ in range(R1.DEPTH):
            nx = self.step(cur)
            if nx is None or nx not in self.cont or nx in seen or not visible(nx): break
            out.append(nx); seen.add(nx); cur = nx
        return out
def run(L, k, raw, lab, sets, ct, phc, wmap):
    path = os.path.join(CK, "%s_%d.pkl" % (L, k))
    if os.path.exists(path): return
    t0 = time.time(); fd = dict(sets)[L][k]; tm, vm = common.split_mask(lab, fd)
    labrec = set(zip(lab.farm, lab.day.astype(int))); val = set(zip(lab.farm[vm], lab.day[vm].astype(int))); trn = set(zip(lab.farm[tm], lab.day[tm].astype(int)))
    dele = labrec - val - trn; raw.set_fold(L, k)
    trT = sorted(trn)
    vis_tr = lambda r: (lambda x: x in raw.train and x not in val and x not in dele)
    stT = []; TR = R1.build_rows(raw, trT, val | dele, True, vis_tr, ct, phc, stT)
    def vis_va(r):
        def v(x):
            if x in dele: return False
            if x in val and (x[0] != r[0] or x[1] >= r[1]): return False
            if x in raw.test and x[0] != r[0]: return False
            return True
        return v
    stV = []; VA = R1.build_rows(raw, sorted(val), dele, False, vis_va, ct, phc, stV)
    w = TR.row_id.map(wmap).values.astype(float); assert np.isfinite(w).all()
    out = VA[["row_id"]].copy()
    for s in SEEDS: out["base_RS_%d" % s] = TF1.fit_base(TR, ct, phc, w, s)(VA)
    out.to_pickle(path); pd.DataFrame(stV).assign(layout=L, fold=k).to_csv(os.path.join(CK, "chains_%s_%d.csv" % (L, k)), index=False)
    print("%s/%d done | TR rows %d VA rows %d | val depth mean %.2f, depth0 %d, c1==prev number %d | %.0fs" % (L, k, len(TR), len(out), np.mean([s["depth"] for s in stV]),
          sum(s["depth"] == 0 for s in stV), sum(s["c1_is_prevnum"] for s in stV), time.time() - t0), flush=True)
def main():
    os.makedirs(CK, exist_ok=True)
    C = json.load(open(os.path.join(EVP.AUD, "cols.json"))); ct, phc = C["ct"], C["phc"]
    L0 = pd.read_pickle(os.path.join(EVP.AUD, "lab_feats.pkl")); wmap = dict(zip(L0.row_id, L0.w))
    lab, pfn, ct2, phc2, wb, wp, wv, sets, z = W.worlds(); assert list(ct2) == list(ct) and list(phc2) == list(phc), "feature list mismatch"
    raw = RawFL(); print("FL links p >= %.1f: %d" % (P_MIN, len(raw.base_link)), flush=True)
    for a in ARGS:
        L, ks = a.split(":")
        for k in map(int, ks.split(",")): run(L, k, raw, lab, sets, ct, phc, wmap)
if __name__ == "__main__":
    main()
