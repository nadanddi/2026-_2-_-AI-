"""IM35b: C3v2 판정 v2(IM_C3v2_v2.md) — 패키지마다 별도 프로세스(predict 그대로), 갈변·밝음은 묶음 단위 판정분, 묶음 짝 부트스트랩. 봉인 두 번째 열람(탐색적)."""
import sys, csv, zlib, json, subprocess, numpy as np
from pathlib import Path
B = Path.home() / "storage/nadan/farmai_img"; S = B / "data/local/sealed_c3"

def run(pk, paths):   # 별도 프로세스(패키지 모듈 캐시 섞임 방지)
    lst = B / "data/local/preds/_paths.json"; out = B / "data/local/preds/_out.npy"; lst.write_text(json.dumps(paths))
    code = ("import sys,json,numpy as np; sys.path.insert(0, %r); import inference as I; m=I.load_model(%r); "
            "np.save(%r, np.array(I.predict(m, json.load(open(%r)))))") % (str(pk), str(pk / "model"), str(out), str(lst))
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True); return np.load(out)

def main():
    R = []
    for sn in ("real", "qwen", "v7", "brown", "bright"):
        for r in csv.DictReader(open(S / sn / "manifest.csv", encoding="utf-8")):
            if sn in ("brown", "bright") and zlib.crc32(r["group"].encode()) % 2 == 0: continue   # 학습에 쓴 묶음 제외
            R.append(dict(r, set=sn, path=str(S / sn / "img" / r["file"])))
    paths = [r["path"] for r in R]; st = np.array([r["set"] for r in R]); grp = np.array([r["group"] for r in R])
    p4 = run(B / "submit_img04/팜모니_나단4", paths); p5 = run(B / "submit_img05/팜모니_나단5", paths)
    np.savez(B / "data/local/preds/sealed_c3v2.npz", p4=p4, p5=p5, set=st, path=np.array(paths))
    r = {s: ((p4[st == s] >= .5).mean(), (p5[st == s] >= .5).mean(), int((st == s).sum())) for s in ("real", "qwen", "v7", "brown", "bright")}
    for s, (a, b, n) in r.items(): print(f"{s:6s} n={n:3d} {'오탐' if s in ('real', 'brown', 'bright') else '탐지'} 나단4 {a:.4f} 나단5 {b:.4f}")
    def pw(sn, better_high):
        idx0 = np.where(st == sn)[0]; gs = np.array(sorted(set(grp[idx0]))); gi = {g: idx0[grp[idx0] == g] for g in gs}; rng = np.random.default_rng(7); bs = []
        for _ in range(2000):
            idx = np.concatenate([gi[g] for g in rng.choice(gs, len(gs))]); d = (p5[idx] >= .5).mean() - (p4[idx] >= .5).mean(); bs.append(d if better_high else -d)
        return float((np.array(bs) <= 0).mean())
    P = {s: pw(s, s in ("qwen", "v7")) for s in ("real", "qwen", "v7", "brown", "bright")}; print("묶음 부트스트랩 p_worse:", {k: round(v, 4) for k, v in P.items()})
    c = [r["bright"][1] <= r["bright"][0], r["brown"][1] <= r["brown"][0] + .01, r["real"][1] <= r["real"][0] + .005, r["v7"][1] >= r["v7"][0] + .03 and P["v7"] < .05, r["qwen"][1] >= r["qwen"][0] - .01]
    print(f"\n== 판정(IM_C3v2_v2.md, 탐색적) ==\n  밝음 {c[0]} · 갈변 {c[1]} · 봉인진짜 {c[2]} · 7판+.03 {c[3]} · Qwen {c[4]} → {'탐색적 후보' if all(c) else '불채택'}")

if __name__ == "__main__":
    main()
