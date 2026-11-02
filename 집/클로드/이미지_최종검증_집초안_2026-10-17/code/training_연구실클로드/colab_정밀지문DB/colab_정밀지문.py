"""Colab용: AI Hub 원천 사진 전체(tar 306개)의 '정밀 지문' 만들기 (IM 정밀 원본대조 준비, 연구실 클로드 10-14).
각 사진 → 배포 형식과 같은 덮게 축소·가운데 크롭(세로 768×1024 기준, 가로는 회전해 세로로 통일) → 두 가지 지문:
  g  : 흑백 24×32 BOX 평균(uint8, 768바이트)  — 국소 편집·합성·스타일 위치 비교용
  e  : 64px 구역(12×16)별 라플라시안 분산 log값(uint8로 양자화, 192바이트) — 블러·노이즈·재그리기 비교용
저장: MyDrive/farmai_finefp/<tar이름>.npz (names, g, e, portrait)  — 이미 있는 tar는 건너뜀(끊겨도 이어서 실행 가능)

사용(Colab, CPU 런타임으로 충분):
  from google.colab import drive; drive.mount('/content/drive')
  !pip -q install opencv-python-headless
  !python "/content/drive/MyDrive/farmai_t4/colab_정밀지문.py"
"""
import os, sys, io, time, tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from PIL import Image

ROOT = Path("/content/drive/MyDrive/농업 AI 경진대회/이미지 미션/099.지능형 수직농장 통합 데이터(딸기)/01.데이터")
OUT = Path("/content/drive/MyDrive/farmai_finefp")

def finefp(data):
    im = Image.open(io.BytesIO(data)); im.draft("RGB", (im.width // 2, im.height // 2))   # JPEG 절반 해상도 디코딩(속도)
    im = im.convert("RGB"); w, h = im.size; port = h >= w
    tw, th = (768, 1024) if port else (1024, 768); s = max(tw / w, th / h)
    nw, nh = max(tw, round(w * s)), max(th, round(h * s)); im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2; a = np.asarray(im.crop((x0, y0, x0 + tw, y0 + th)).convert("L"), np.float32)
    if not port: a = np.ascontiguousarray(np.rot90(a))   # 세로 768×1024로 통일
    g = cv2.resize(a, (24, 32), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(a, cv2.CV_32F, ksize=3); v = lap.reshape(16, 64, 12, 64).var(axis=(1, 3))
    e = np.clip(np.round((np.log1p(v) / 10.0) * 255), 0, 255)
    return np.clip(np.round(g), 0, 255).astype(np.uint8), e.astype(np.uint8), port

def one_tar(tp):
    out = OUT / f"{tp.stem}.npz"
    if out.exists(): return tp.stem, -1, 0.0
    t = time.time(); names, G, E, P = [], [], [], []
    with tarfile.open(tp, "r|") as tf:
        for m in tf:
            if not (m.isfile() and m.name.lower().endswith(".jpg")): continue
            try:
                g, e, p = finefp(tf.extractfile(m).read())
            except Exception:
                continue
            names.append(Path(m.name).stem); G.append(g); E.append(e); P.append(p)
    tmp = OUT / f"{tp.stem}.tmp.npz"
    np.savez_compressed(tmp, names=np.array(names), g=np.stack(G), e=np.stack(E), portrait=np.array(P)); os.replace(tmp, out)
    return tp.stem, len(names), time.time() - t

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tars = sorted((ROOT / "2.Validation" / "원천데이터").glob("*.tar")) + sorted((ROOT / "1.Training" / "원천데이터_add_20260807").glob("*.tar"))
    todo = [t for t in tars if not (OUT / f"{t.stem}.npz").exists()]
    print(f"tar 전체 {len(tars)}개, 남은 {len(todo)}개", flush=True)
    if not tars: sys.exit("tar를 찾지 못함: ROOT 경로를 확인하세요 → " + str(ROOT))
    t0 = time.time(); done = 0; n_img = 0
    with ProcessPoolExecutor(max(2, os.cpu_count() or 2)) as ex:
        for stem, n, sec in ex.map(one_tar, todo):
            done += 1; n_img += max(n, 0)
            el = time.time() - t0; eta = el / done * (len(todo) - done)
            print(f"[{done}/{len(todo)}] {stem} {n}장 {sec:.0f}s | 누적 {n_img}장, 경과 {el / 60:.1f}분, 남은 예상 {eta / 60:.0f}분", flush=True)
    print("FINEFP_DONE", n_img)

if __name__ == "__main__":
    main()
