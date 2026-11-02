"""IM28: G3 지름길 반증용 시험 세트 3종 (IM_G3_지름길반증_v1.md). 로컬 실행 → local/im_set_gpt3r, im_set_realn, im_set_realg
v3R: GPT v3 PNG → AI허브 원본 긴 변 분포로 확대(LANCZOS) → JPEG Q90 → 배포 처리
realN: 원본2·3 진짜 → 배포 처리 / realG: 같은 진짜 → GPT v3 출력 긴 변 분포로 축소 → PNG(무손실) → 배포 처리
"""
import csv, io, random, zlib
from pathlib import Path
from PIL import Image
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
V3 = Path(__file__).resolve().parents[2] / "코덱스/local/ai_gen_work_20261014_v3"; AG = LOCAL / "ai_gen_work"

def resize_long(im, L):
    s = L / max(im.size); return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)

def main():
    reals = []
    for lst, d in [("원본2_목록.csv", "원본2"), ("원본3_목록.csv", "원본3")]:
        reals += [(AG / d / r["파일명"], r["AI허브_원천"]) for r in csv.DictReader(open(AG / lst, encoding="utf-8-sig"))]
    real_long = [max(Image.open(p).size) for p, _ in reals]
    gpts = sorted((V3 / "결과_GPT/3_부분생성").glob("*.png")) + sorted((V3 / "결과_GPT/4_부분스타일").glob("*.png")) + sorted((V3 / "중간_GPT").glob("*.png"))
    gpt_long = [max(Image.open(p).size) for p in gpts]
    v3m = {r["info"].split(";")[1]: r for r in csv.DictReader(open(LOCAL / "im_set_gpt3/manifest.csv", encoding="utf-8"))}
    def write(sn, items):
        out = LOCAL / sn; (out / "img").mkdir(parents=True, exist_ok=True)
        with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]); w.writeheader()
            for name, img, row in items: (out / "img" / name).write_bytes(jpeg_bytes(img)); w.writerow(row)
        print(sn, len(items))
    it = []
    for p in gpts:
        rng = random.Random(zlib.crc32(("R|" + p.name).encode())); im = resize_long(Image.open(p).convert("RGB"), rng.choice(real_long))
        b = io.BytesIO(); im.save(b, "JPEG", quality=90); im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        base = v3m[p.name]; name = "r_" + base["file"]
        it.append((name, dist_cover(im, rng), dict(base, file=name, info=base["info"] + ";camera_path")))
    write("im_set_gpt3r", it)
    for sn, mode in [("im_set_realn", "N"), ("im_set_realg", "G")]:
        it = []
        for k, (p, src) in enumerate(reals):
            rng = random.Random(zlib.crc32((mode + "|" + p.name).encode())); im = Image.open(p).convert("RGB")
            if mode == "G":
                im = resize_long(im, rng.choice(gpt_long)); b = io.BytesIO(); im.save(b, "PNG"); im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
            stem = Path(src.replace("\\", "/")).stem; name = f"{mode.lower()}_{p.stem}.jpg"
            it.append((name, dist_cover(im, rng), dict(file=name, label=0, kind="real_" + mode.lower(), info=f"{mode}_path;{p.name}", source=src,
                                                        group="_".join(stem.split("_")[:3]), farm_code=stem.split("_")[0], tar=src.replace("\\", "/").split("/")[0], other=p.name)))
        write(sn, it)

if __name__ == "__main__":
    main()
