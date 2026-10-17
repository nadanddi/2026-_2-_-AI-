"""IM27: 팀원 제출 패키지(submission_package.zip, ConvNeXt-Tiny 384) 추론 재현 + 우리 제출 01과 같은 사진에서 비교. 서버 실행.
팀원 inference.py와 같은 전처리(Resize 384×384, ImageNet 정규화, 좌우 반전 평균). 가중치는 weights_only=True로 읽음(팀원 코드는 실행하지 않음).
평가 사진(양쪽 모두 학습에 안 쓴 것 기준):
 - 공개 21장
 - R2 보류 자료(im_set_r2: 우리 최종 학습에 안 씀. 진짜 + 우리 방식 가짜)
 - GPT v1·v2 250장, GPT v3 최종·1단계(진행 중인 만큼), 원본3 진짜 100장
출력: logs/teammate3_cmp.log
"""
import sys, csv, json, numpy as np, torch, timm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
B = Path.home() / "storage/nadan/farmai_img"; LOCAL = B / "data/local"; PKG = B / "ext/teammate_pkg3"
PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"

def tm_model():
    m = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    sd = torch.load(PKG / "model/weights.pt", map_location="cpu", weights_only=True)
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}; m.load_state_dict(sd); return m.cuda().eval()

TF = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def tm_pred(m, paths):
    out = []
    for i in range(0, len(paths), 16):
        x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + 16]]).cuda()
        out += torch.sigmoid((m(x).squeeze(-1) + m(torch.flip(x, dims=[3])).squeeze(-1)) / 2).cpu().tolist()
    return np.array(out)

def ours_pred(paths):
    sys.path.insert(0, str(B / "submit_img01/팜모니_나단1")); import inference as I  # noqa
    mm = I.load_model(); return np.array(I.predict(mm, [str(p) for p in paths]))

def sets():
    S = []
    S += [("공개21", PUB / "images" / r["file"], int(r["label"]), r["type"]) for r in csv.DictReader(open(PUB / "sample_labels.csv"))]
    S += [("R2보류", LOCAL / "im_set_r2/img" / r["file"], int(r["label"]), r["kind"]) for r in csv.DictReader(open(LOCAL / "im_set_r2/manifest.csv", encoding="utf-8"))]
    for sn in ("im_set_gpt1", "im_set_gpt2"):
        S += [("GPT v1v2", LOCAL / sn / "img" / r["file"], 1, r["kind"]) for r in csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8"))]
    q = B / "q_v3"
    S += [("GPT v3 최종", p, 1, "v3_" + p.stem.split("_")[2]) for p in sorted((q / "F").glob("*.jpg"))]
    S += [("GPT v3 1단계", p, 1, "v3_stage1") for p in sorted((q / "S").glob("*.jpg"))]
    S += [("원본3 진짜", p, 0, "real") for p in sorted((q / "O").glob("*.jpg"))]
    return S

def main():
    S = sets(); paths = [s[1] for s in S]; y = np.array([s[2] for s in S]); grp = np.array([s[0] for s in S]); kd = np.array([s[3] for s in S])
    pt = tm_pred(tm_model(), paths); po = ours_pred(paths)
    np.savez(LOCAL / "preds/teammate3_cmp.npz", grp=grp, kind=kd, y=y, pt=pt, po=po, files=np.array([str(p) for p in paths]))
    for g in ["공개21", "R2보류", "GPT v1v2", "GPT v3 최종", "GPT v3 1단계", "원본3 진짜"]:
        s = grp == g; acc = lambda p: ((p[s] >= .5) == y[s]).mean()
        line = f"{g:12s} n={s.sum():4d} 정확도 팀원 {acc(pt):.3f} 우리 {acc(po):.3f}"
        if len(set(y[s])) == 2:
            r = s & (y == 0); f = s & (y == 1)
            line += f" | 진짜오탐 팀원 {(pt[r] >= .5).mean():.3f} 우리 {(po[r] >= .5).mean():.3f} | 가짜탐지 팀원 {(pt[f] >= .5).mean():.3f} 우리 {(po[f] >= .5).mean():.3f}"
            line += f" | AUC 팀원 {roc_auc_score(y[s], pt[s]):.3f} 우리 {roc_auc_score(y[s], po[s]):.3f}"
        print(line, flush=True)
        if g in ("공개21", "R2보류", "GPT v1v2", "GPT v3 최종"):
            for k in sorted(set(kd[s])):
                t = s & (kd == k); print(f"     {k:18s} n={t.sum():3d} '정답' 비율 팀원 {((pt[t] >= .5) == y[t]).mean():.2f} 우리 {((po[t] >= .5) == y[t]).mean():.2f}", flush=True)
    s = grp == "공개21"; print("공개 개별(팀원/우리):", " ".join(f"{Path(f).stem}({k[:5]},{l})={a:.2f}/{b:.2f}" for f, k, l, a, b in zip(np.array(paths)[s], kd[s], y[s], pt[s], po[s])))

if __name__ == "__main__":
    main()
