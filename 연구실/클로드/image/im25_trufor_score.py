"""IM25: TruFor(GRIP-UNINA, 공개 가중치) 무학습 점수 — 부분 편집 전용 모델 후보 탐색(사전 시험, 판정 아님).
대상: 진짜(v4·sd2·v5 매니페스트의 real 전부), GPT 부분편집 v1·v2 전부, 6판 무작위 300, 5판 partial_style_xl 무작위 300, 공개 21.
출력: local/feats/trufor_scores.npz (set, file, det=이미지 점수, mmax=위치맵 최댓값, m99=위치맵 상위1% 평균, mmean)
라이선스: 비영리·정보 목적 한정(LICENSE.txt). 대회 제출 사용 가능 여부는 별도 확인 전까지 미정.
"""
import sys, csv, time, numpy as np, torch
from pathlib import Path
from PIL import Image
TF = Path.home() / "storage/nadan/farmai_img/ext/TruFor/test_docker"
sys.path.insert(0, str(TF / "src"))
from config import _C as cfg  # noqa
LOCAL = Path(__file__).resolve().parents[1] / "local"
PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"

def model():
    cfg.defrost(); cfg.merge_from_file(str(TF / "src/trufor.yaml")); cfg.TEST.MODEL_FILE = str(TF / "weights/trufor.pth.tar"); cfg.freeze()
    from models.cmx.builder_np_conf import myEncoderDecoder
    m = myEncoderDecoder(cfg=cfg); m.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location="cpu")["state_dict"]); return m.cuda().eval()

def jobs():
    rng = np.random.default_rng(0); out = []
    for sn in ("im_set_v4", "im_set_sd2", "im_set_v5"):
        out += [(sn, LOCAL / sn / "img" / r["file"]) for r in csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8")) if r["kind"] == "real"]
    for sn in ("im_set_gpt1", "im_set_gpt2"):
        out += [(sn, LOCAL / sn / "img" / r["file"]) for r in csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8")) if r["kind"] in ("gpt_pgen", "gpt_pstyle")]
    v6 = [r["file"] for r in csv.DictReader(open(LOCAL / "im_set_v6/manifest.csv", encoding="utf-8"))]; out += [("im_set_v6", LOCAL / "im_set_v6/img" / f) for f in rng.choice(v6, 300, replace=False)]
    v5 = [r["file"] for r in csv.DictReader(open(LOCAL / "im_set_v5/manifest.csv", encoding="utf-8")) if r["kind"] == "partial_style_xl"]
    out += [("im_set_v5", LOCAL / "im_set_v5/img" / f) for f in rng.choice(v5, 300, replace=False)]
    out += [("public21", PUB / "images" / r["file"]) for r in csv.DictReader(open(PUB / "sample_labels.csv"))]
    return out

@torch.no_grad()
def main():
    m = model(); J = jobs(); res = []; t = time.time()
    for n, (sn, p) in enumerate(J):
        x = torch.tensor(np.array(Image.open(p).convert("RGB")).transpose(2, 0, 1), dtype=torch.float)[None].cuda() / 256.0
        pred, conf, det, _ = m(x); mp = torch.softmax(pred, 1)[0, 1].cpu().numpy(); v = np.sort(mp.ravel())
        res.append((sn, p.name, float(torch.sigmoid(det).item()), float(v[-1]), float(v[-max(1, len(v) // 100):].mean()), float(v.mean())))
        if n % 500 == 0: print(n, len(J), f"{time.time()-t:.0f}s", flush=True)
    a = np.array(res, dtype=object)
    np.savez(LOCAL / "feats" / "trufor_scores.npz", set=a[:, 0].astype(str), fname=a[:, 1].astype(str), det=a[:, 2].astype(float), mmax=a[:, 3].astype(float), m99=a[:, 4].astype(float), mmean=a[:, 5].astype(float))
    print("saved", len(res), f"{time.time()-t:.0f}s ({(time.time()-t)/len(res):.3f}s/장)", flush=True)

if __name__ == "__main__":
    main()
