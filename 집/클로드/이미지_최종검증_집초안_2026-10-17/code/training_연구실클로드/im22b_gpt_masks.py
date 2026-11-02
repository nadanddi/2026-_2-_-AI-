"""IM22b: GPT 부분 편집(v1·v2)의 편집 위치 마스크 = 원본과의 차이 (IM_국소패치_L_v1.md). 로컬 실행.
배포 형식 GPT 사진과, 같은 크기로 덮게 축소·가운데 크롭한 원본을 비교:
 Lab 차이 → 가우시안(σ=3) → 기준 = max(중앙값 + 5·MAD, 15) 초과 → 열기·닫기 → 넓이 .05% 미만 덩어리 제거.
GPT는 사진 전체를 약하게 다시 그리므로 전역 차이는 중앙값·MAD로 걷어낸다. 넓이가 50%를 넘거나 .1% 미만이면 '신뢰 불가'(조각 학습 제외).
출력: local/feats/im_set_gpt{1,2}_pmask.npz (files, ov=(N,54), area, ok), 점검용 그림 local/feats/gpt_mask_check/*.jpg
"""
import csv, numpy as np, cv2
from pathlib import Path
from PIL import Image
from im03_make_fakes import LOCAL  # noqa
from im22_patch_masks import overlap  # noqa
AG = LOCAL / "ai_gen_work"

def lab(a): return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)

def main():
    (LOCAL / "feats/gpt_mask_check").mkdir(parents=True, exist_ok=True)
    for sn, od in [("im_set_gpt1", AG / "원본"), ("im_set_gpt2", AG / "원본2")]:
        files, ovs, areas, oks = [], [], [], []
        for k, r in enumerate(csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8"))):
            if r["kind"] not in ("gpt_pgen", "gpt_pstyle"): continue
            g = np.asarray(Image.open(LOCAL / sn / "img" / r["file"]).convert("RGB")); h, w = g.shape[:2]
            o = Image.open(od / r["other"]).convert("RGB"); s = max(w / o.width, h / o.height)
            o = o.resize((max(w, round(o.width * s)), max(h, round(o.height * s))), Image.LANCZOS); x0, y0 = (o.width - w) // 2, (o.height - h) // 2
            o = np.asarray(o.crop((x0, y0, x0 + w, y0 + h)))
            d = cv2.GaussianBlur(np.linalg.norm(lab(g) - lab(o), axis=2), (0, 0), 3)
            med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6; m = (d > max(med + 5 * 1.4826 * mad, 15)).astype(np.uint8)
            k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)); m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k9), cv2.MORPH_CLOSE, k9)
            n, labm, st, _ = cv2.connectedComponentsWithStats(m, 8)
            m = np.isin(labm, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= .0005 * m.size]).astype(np.float32)
            area = float(m.mean()); ok = .001 <= area <= .5
            files.append(r["file"]); ovs.append(overlap(m)); areas.append(area); oks.append(ok)
            if True:
                vis = g.copy(); vis[m > 0] = (vis[m > 0] * .5 + np.array([255, 0, 0]) * .5).astype(np.uint8)
                Image.fromarray(np.concatenate([o, vis], 1)).resize((w, h // 2)).save(LOCAL / "feats/gpt_mask_check" / f"{sn}_{r['file']}")
        np.savez(LOCAL / "feats" / f"{sn}_pmask.npz", files=np.array(files), ov=np.stack(ovs), area=np.array(areas), ok=np.array(oks))
        a = np.array(areas); print(sn, len(files), "신뢰", int(np.sum(oks)), "넓이 중앙값", round(float(np.median(a)), 3), "분위(10,90)", np.round(np.quantile(a, [.1, .9]), 3),
                                    "양성조각 평균(신뢰분)", round(float((np.stack(ovs)[np.array(oks)] >= .25).sum(1).mean()), 1))

if __name__ == "__main__":
    main()
