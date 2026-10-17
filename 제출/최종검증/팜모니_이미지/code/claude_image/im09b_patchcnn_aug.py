"""IM09b: 후보 C7 = IM09 + 공통 처리 증강 (단일 변경, 사용자 지시 10-11).
 증강: 쌍(가짜·짝 실제)에 같은 매개변수로 적용. 75% 확률로 [0.75~1.0 축소 → 밝기·대비 ±8% → JPEG 88~95 재저장], 25% 무처리(팀원 v5 설정).
 나머지(에폭·K·폴드·집계)는 IM09와 동일. 출력 preds/cnn_v1aug.npz
원본 설명: IM09: 원해상도 패치 CNN 갈래 (사용자 지시 10-11 '3번', 합성 경계·부분 편집용).
구조(팀원 v5와 같은 뼈대): 고정 고주파 잔차 (x - GaussBlur5(σ1))*8 → ResNet18(ImageNet 초기화) → 패치 로짓.
학습(우리 방식):
 - 쌍 학습: 가짜 이미지와 같은 원본의 실제 이미지에서 **같은 좌표** 패치 K개씩 → 장면이 아닌 조작 흔적만 차이.
   원본이 없는 ai_gen은 무작위 실제와 짝.
 - 이미지 점수 = 패치 로짓 상위 2개 평균(MIL), BCE. SD 부분 생성·부분 스타일은 manifest의 tile 좌표 안에서 70% 패치 추출.
 - 증강: 좌우·상하 반전(쌍에 동일 적용)만. 블러·잡음류 증강 금지(가짜 흔적과 겹침).
 - 학습 가중: im05c.weights(실제 .5 / 공식 6유형 균등).
추론: cover 격자(4x5=20 패치), 패치 확률 상위 5개 평균(v5와 같은 고정 규칙, 공개 21장으로 고르지 않음).
검증: farm_code 5폴드(im05c.folds 시드 0) OOF + 전체 학습 모델로 공개 21장·빨간 조명 행 예측 → local/preds/cnn_v1.npz
사용: python im09_patchcnn.py [epochs=3] [K=6]
"""
import sys, csv, math, random, time, re, io, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import models
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as E  # noqa

LOCAL = E.LOCAL; PUB = E.PUB; SETS = ["im_set_v4", "im_set_sd2"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class HighPass(nn.Module):
    def __init__(s, sigma=1.0, size=5, gain=8.0):
        super().__init__(); r = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-r ** 2 / (2 * sigma ** 2)); g /= g.sum()
        s.register_buffer("k", (g[:, None] * g[None, :]).expand(3, 1, size, size).clone(), persistent=False); s.p, s.gain = size // 2, gain
    def forward(s, x): return (x - F.conv2d(F.pad(x, [s.p] * 4, mode="reflect"), s.k, groups=3)) * s.gain

class Net(nn.Module):
    def __init__(s, pretrained=True):
        super().__init__(); b = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None); b.fc = nn.Identity()
        s.pre, s.b, s.head = HighPass(), b, nn.Linear(512, 1)
    def forward(s, x): return s.head(s.b(s.pre(x))).squeeze(1)

def load_img(p): return np.asarray(Image.open(p).convert("RGB"))
def norm(patches): x = (np.stack(patches).astype(np.float32) / 255 - MEAN) / STD; return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())

def rows_all():
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}; R = []
    for s in SETS:
        for r in csv.DictReader(open(LOCAL / s / "manifest.csv", encoding="utf-8")):
            if "tile_whole=1" in r["info"] or r["kind"] not in E.TYPE: continue
            R.append(dict(r, set=s, red=r["source"] in red))
    return R

def common_proc(img, sc, br, ct, q):
    im = Image.fromarray(img)
    if sc < .999: im = im.resize((max(224, round(im.width * sc)), max(224, round(im.height * sc))), Image.BICUBIC)
    x = np.asarray(im, np.float32); x = np.clip((x - 128) * ct + 128 + br * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(x).save(buf, "JPEG", quality=q, subsampling=2); return np.asarray(Image.open(buf).convert("RGB"))

def coords(h, w, K, rng, tile=None):
    out = []
    for _ in range(K):
        if tile is not None and rng.random() < .7:
            x0, y0 = tile; x = rng.randint(min(x0, w - 224), max(min(x0, w - 224), min(x0 + 512 - 224, w - 224))); y = rng.randint(min(y0, h - 224), max(min(y0, h - 224), min(y0 + 512 - 224, h - 224)))
        else: x, y = rng.randint(0, w - 224), rng.randint(0, h - 224)
        out.append((y, x))
    return out

def batch_pairs(items, real_of, R, rng, K):
    P, lab, n_img = [], [], 0
    for i in items:
        r = R[i]; a = load_img(LOCAL / r["set"] / "img" / r["file"]); h, w = a.shape[:2]
        mt = re.search(r"tile=(\d+),(\d+)", r["info"]); tile = (int(mt.group(1)), int(mt.group(2))) if mt else None
        j = real_of.get(r["source"]) if r["label"] == "1" else None
        if r["label"] == "1" and j is None: j = rng.choice(real_of["__any__"])
        b = load_img(LOCAL / R[j]["set"] / "img" / R[j]["file"]) if j is not None else None
        if b is not None and b.shape != a.shape: b = None if r["kind"] != "ai_gen" else b
        if rng.random() < .75:   # 공통 처리 증강(쌍 동일)
            sc, br, ct, q = rng.uniform(.75, 1.0), rng.uniform(-.08, .08), rng.uniform(.92, 1.08), rng.randint(88, 95)
            a = common_proc(a, sc, br, ct, q)
            if b is not None: b = common_proc(b, sc, br, ct, q)
            h, w = a.shape[:2]
            if tile is not None: tile = (int(tile[0] * sc), int(tile[1] * sc))
            if tile is not None and (tile[0] + 512 * sc > w or 512 * sc < 224): tile = None
        cs = coords(h, w, K, rng, tile)
        fl = (rng.random() < .5, rng.random() < .5)
        def cut(img, y, x):
            p = img[y:y + 224, x:x + 224] if y + 224 <= img.shape[0] and x + 224 <= img.shape[1] else img[:224, :224]
            if fl[0]: p = p[:, ::-1]
            if fl[1]: p = p[::-1]
            return p
        P += [cut(a, y, x) for y, x in cs]; lab.append(int(r["label"])); n_img += 1
        if b is not None:
            hb, wb = b.shape[:2]
            P += [cut(b, min(y, hb - 224), min(x, wb - 224)) for y, x in cs]; lab.append(0); n_img += 1
    return norm(P), torch.tensor(lab, dtype=torch.float32), n_img

def train(tr, R, w, epochs, K, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    real_of = {R[i]["source"]: i for i in tr if R[i]["kind"] == "real"}; real_of["__any__"] = [i for i in tr if R[i]["kind"] == "real"]
    m = Net().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
    fakes = [i for i in tr if R[i]["label"] == "1"]; reals_un = [i for i in tr if R[i]["label"] == "0" and R[i]["kind"] != "real"]
    pool = fakes + reals_un   # 실제는 가짜의 쌍으로 들어가고, 리샘플 실제는 단독
    steps = epochs * math.ceil(len(pool) / 4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps)
    pw = {i: w[k] for k, i in enumerate(tr)} if False else None
    m.train(); t = time.time(); st = 0
    for ep in range(epochs):
        rng.shuffle(pool)
        for bi in range(0, len(pool), 4):
            x, y, n = batch_pairs(pool[bi:bi + 4], real_of, R, rng, K)
            logit = m(x.to(DEV)).view(n, K); s = logit.topk(2, 1).values.mean(1)
            loss = F.binary_cross_entropy_with_logits(s, y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); st += 1
            if st % 200 == 0: print(f"   ep{ep} step {st}/{steps} loss {loss.item():.3f} {time.time()-t:.0f}s", flush=True)
    return m.eval()

@torch.no_grad()
def predict(m, paths, k=5):
    out = []
    for p in paths:
        a = load_img(p); h, w = a.shape[:2]
        xs = np.linspace(0, w - 224, math.ceil(w / 224)).round().astype(int); ys = np.linspace(0, h - 224, math.ceil(h / 224)).round().astype(int)
        pp = torch.sigmoid(m(norm([a[y:y + 224, x:x + 224] for y in ys for x in xs]).to(DEV)))
        out.append(float(pp.topk(min(k, len(pp))).values.mean()))
    return np.array(out)

def main(epochs=3, K=6):
    R = rows_all(); keep = [i for i, r in enumerate(R) if not r["red"]]; Rk = [R[i] for i in keep]
    kind = np.array([r["kind"] for r in Rk]); w = E.weights(kind); grp = np.array([r["farm_code"] for r in Rk])
    print(f"학습 행 {len(Rk)}, 장치 {DEV}", flush=True)
    oof = np.full(len(Rk), np.nan)
    for f, (tr, te) in enumerate(E.folds(grp, 0)):
        print(f"fold {f}: train {len(tr)} test {len(te)}", flush=True)
        m = train(list(tr), Rk, w, epochs, K, f)
        oof[te] = predict(m, [LOCAL / Rk[i]["set"] / "img" / Rk[i]["file"] for i in te]); del m; torch.cuda.empty_cache()
    print("전체 학습", flush=True); m = train(list(range(len(Rk))), Rk, w, epochs, K, 99)
    lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pub = predict(m, [PUB / "images" / r["file"] for r in lab])
    red_rows = [R[i] for i in range(len(R)) if R[i]["red"]]
    # 빨간 조명 진단은 im_set_v2의 빨간 행(가짜 v3/v4에는 없음)
    v2 = [r for r in csv.DictReader(open(LOCAL / "im_set_v2" / "manifest.csv", encoding="utf-8"))
          if r["source"] in set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) and r["kind"] in E.TYPE]
    redp = predict(m, [LOCAL / "im_set_v2" / "img" / r["file"] for r in v2])
    (LOCAL / "preds").mkdir(exist_ok=True)
    np.savez(LOCAL / "preds" / "cnn_v1aug.npz", files=np.array([f"{r['set']}/{r['file']}" for r in Rk]), oof=oof,
             pub_files=np.array([r["file"] for r in lab]), pub=pub, red_files=np.array([f"im_set_v2/{r['file']}" for r in v2]), red=redp)
    torch.save(m.state_dict(), LOCAL / "preds" / "cnn_v1aug_full.pt")
    py = np.array([int(r["label"]) for r in lab])
    print(f"공개21 acc={((pub >= .5) == py).mean():.3f} ({((pub >= .5) == py).sum()}/21) | " + " ".join(f"{r['file'][:4]}({r['type'][:5]})={p:.2f}" for r, p in zip(lab, pub)))
    y = np.array([int(r["label"]) for r in Rk]); t = np.array([E.TYPE[k] for k in kind])
    print(f"OOF 균형acc={E.bal_acc(oof, y, t):.3f} | " + " ".join(f"{x}={((oof[t == x] >= .5) == (x != 'real')).mean():.2f}" for x in ["real"] + E.TYPES6))
    yr = np.array([int(r["label"]) for r in v2]); print(f"빨강 실제정답={(redp[yr == 0] < .5).mean():.2f} 빨강가짜탐지={(redp[yr == 1] >= .5).mean():.2f}")

if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
