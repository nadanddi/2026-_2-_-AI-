"""IM11: AI허브 전체(원천 TS·VS tar) 지문 DB — 원본 대조 가설용.
근거(IM10): 공개 실제 qube와 원본의 6x8 색 지문 차 0.11, 부분 생성 uexm과 그 원본 13.8, 이웃 프레임 27.4.
지문: 배포 처리와 같은 '덮게 축소 → 가운데 크롭'(768x1024 비율) 후 BOX 평균으로 12x16(세로 기준) RGB → uint8.
      가로 사진은 16x12로 계산한 뒤 저장 시 (세로 기준) 회전 없이 별도 방향 플래그로 둠.
속도: PIL draft로 JPEG를 1/8 크기로 디코딩(평균 지문에는 영향이 작음). tar 단위 병렬(프로세스 4).
출력: local/fp_db/<tar>.npz (names, fp(12x16x3 uint8), portrait, w, h) — 재개 가능(이미 있으면 건너뜀).
"""
import sys, tarfile, io, time, numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(r"G:\내 드라이브\농업 AI 경진대회\이미지 미션\099.지능형 수직농장 통합 데이터(딸기)\01.데이터")
OUT = Path(__file__).resolve().parents[1] / "local" / "fp_db"

def fingerprint(img):
    w, h = img.size; port = h >= w; tw, th = (768, 1024) if port else (1024, 768)
    s = max(tw / w, th / h); nw, nh = w * s, h * s; x0, y0 = (nw - tw) / 2 / s, (nh - th) / 2 / s
    c = img.crop((round(x0), round(y0), round(x0 + tw / s), round(y0 + th / s)))
    fp = np.asarray(c.convert("RGB").resize((12, 16) if port else (16, 12), Image.BOX), np.uint8)
    return fp, port

def fp_bytes(data):
    im = Image.open(io.BytesIO(data)); W, H = im.size; im.draft("RGB", (W // 8, H // 8))
    # draft 뒤 크기가 줄어도 비율은 같음 → 원래 W,H 기준 크롭 비율 유지
    fp, port = fingerprint(im); return fp, port, W, H

def do_tar(tp):
    out = OUT / f"{tp.stem}.npz"
    if out.exists(): return tp.stem, -1, 0
    t = time.time(); names, fps, ports, ws, hs = [], [], [], [], []
    with tarfile.open(tp, "r|") as tf:   # 순차 스트리밍(색인 없이 한 번 읽기)
        for m in tf:
            if not (m.isfile() and m.name.lower().endswith(".jpg")): continue
            try:
                fp, port, W, H = fp_bytes(tf.extractfile(m).read())
            except Exception:
                continue
            names.append(Path(m.name).stem); fps.append(fp if port else fp.transpose(1, 0, 2)); ports.append(port); ws.append(W); hs.append(H)
    np.savez_compressed(out, names=np.array(names), fp=np.stack(fps), portrait=np.array(ports), w=np.array(ws), h=np.array(hs))
    return tp.stem, len(names), time.time() - t

def main(workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    tars = sorted((ROOT / "2.Validation" / "원천데이터").glob("*.tar")) + sorted((ROOT / "1.Training" / "원천데이터_add_20260807").glob("*.tar"))
    print("tar", len(tars), flush=True); t0 = time.time(); n = 0
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(do_tar, tp) for tp in tars]):
            name, k, dt = f.result(); n += max(k, 0)
            print(f"{name}: {k}장 {dt:.0f}s | 누적 {n}장 {time.time()-t0:.0f}s", flush=True)
    print("done")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
