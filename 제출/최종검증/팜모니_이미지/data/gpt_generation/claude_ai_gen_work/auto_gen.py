"""GPT·Gemini 가짜 자동 생성: API로 만들고, 결과_<도구>/<유형>/에 바로 저장하고, 기록.csv에 한 줄씩 적는다.
API 키는 사용자가 직접 환경변수로 넣는다(OPENAI_API_KEY / GEMINI_API_KEY). 이 스크립트는 키를 저장하거나 출력하지 않는다.
사용:
  python auto_gen.py --tool gemini --n 5            # 유형마다 5장(먼저 소량으로 확인)
  python auto_gen.py --tool gpt --n 30
  python auto_gen.py --tool gemini --n 30 --dry     # API 호출 없이 계획만 출력
같은 결과 파일이 이미 있으면 건너뛴다(중단 후 다시 실행해도 이어서 진행).
"""
import argparse, base64, csv, io, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = sorted((ROOT / "원본").glob("src_*.jpg"))

GEN = [  # 1 전체생성 (원본 없음)
    "A realistic smartphone photo inside a strawberry vertical farm: strawberry plants in white hydroponic gutters, black drip irrigation tubes hanging, pebbles on the floor, natural indoor lighting. Photorealistic, no people, no text.",
    "A realistic close-up smartphone photo of strawberry plants on a raised greenhouse bench, with {stage} visible and a yellow sticky insect trap among the leaves. Photorealistic texture, no people, no text.",
    "A realistic side-view photo of a row of strawberry seedlings in black growing containers, greenhouse frame and translucent roof behind, overcast natural light. No people, no text.",
    "A realistic phone photo of a strawberry vertical farm with rows of white plastic pots, small plant labels, leaves drooping down, white LED lighting. No people.",
    "A realistic top-down photo of strawberry plants growing in white coco-peat grow bags, substrate visible between the leaves. Photorealistic, no people, no text.",
    "A realistic close-up photo of strawberry runners hanging over the edge of a growing bench, blurred greenhouse interior in the background. No people.",
    "A realistic portrait-orientation close-up of strawberry leaves with sharp veins and serrated edges, a large yellow sticky trap behind them. Photorealistic.",
    "A realistic wide-angle smartphone photo taken from the aisle of a strawberry greenhouse, dense strawberry plants on growing benches on both sides. No people.",
]
STAGES = ["white strawberry flowers", "small green strawberries", "ripe red strawberries"]
FSTYLE = ["watercolor painting", "oil painting", "colored pencil drawing", "digital illustration", "storybook illustration",
          "anime background art", "pastel drawing", "smooth 3D render", "realistic illustration with clean flat color areas"]
PGEN = [
    "Edit only the {pos} area of this photo: add {obj} naturally among the leaves. Keep everything else exactly unchanged and photorealistic, matching the original lighting and colors.",
    "Edit only the {pos} area of this photo: replace the plant there with a similar but different strawberry plant. Keep everything else exactly unchanged, same lighting and color.",
    "Remove one {thing} from the {pos} area of this photo and fill the area naturally. Keep everything else exactly unchanged and photorealistic.",
]
POS = ["lower-right", "upper-left", "center", "lower-left", "upper-right"]
OBJ = ["a white strawberry flower", "a ripe red strawberry", "a small green strawberry", "a few new leaves"]
THING = ["dry or wilted leaf", "plant label", "piece of tubing", "yellow sticky trap"]
PSTYLE = [
    "Change only the {part} into a {style} style; keep the rest of the photo (background, facilities, floor) exactly as it is.",
    "Make only the {part} look painterly, changing their texture and shading like a {style}; keep everything else photographically unchanged.",
]
PART = ["central plant", "plant on the right", "leaves in the foreground", "plant on the left"]
PSTY = ["oil painting", "watercolor", "illustration", "digital painting"]

TYPES = {"1_전체생성": "gen", "2_전체스타일": "fstyle", "3_부분생성": "pgen", "4_부분스타일": "pstyle"}

def make_jobs(n, tool, seed):
    rnd = random.Random(seed + (0 if tool == "gpt" else 1)); jobs = []
    srcs = SRC[:]; rnd.shuffle(srcs); k = 0
    for typ, code in TYPES.items():
        for i in range(n):
            if code == "gen":
                p = rnd.choice(GEN).format(stage=rnd.choice(STAGES)); jobs.append((typ, None, p, f"g{i+1:03d}_gen.png")); continue
            src = srcs[k % len(srcs)]; k += 1
            if code == "fstyle":
                st = rnd.choice(FSTYLE); p = f"Keep the exact composition and object positions of this photo, but redraw the whole image as a {st}."; memo = st.split()[0]
            elif code == "pgen":
                p = rnd.choice(PGEN).format(pos=rnd.choice(POS), obj=rnd.choice(OBJ), thing=rnd.choice(THING)); memo = "pgen"
            else:
                p = rnd.choice(PSTYLE).format(part=rnd.choice(PART), style=rnd.choice(PSTY)); memo = "pstyle"
            jobs.append((typ, src, p, f"{src.stem}_{memo}_{i+1:03d}.png"))
    return jobs

def run_gpt(prompt, src, model):
    from openai import OpenAI
    c = OpenAI()
    if src is None:
        r = c.images.generate(model=model, prompt=prompt, size="1024x1536")
    else:
        with open(src, "rb") as f:
            r = c.images.edit(model=model, image=f, prompt=prompt, size="1024x1536")
    return base64.b64decode(r.data[0].b64_json)

def run_gemini(prompt, src, model):
    from google import genai
    from PIL import Image
    c = genai.Client()
    contents = [prompt] if src is None else [prompt, Image.open(src)]
    r = c.models.generate_content(model=model, contents=contents)
    for part in r.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return part.inline_data.data
    raise RuntimeError("이미지 없이 글만 돌아옴: " + (r.text or "")[:120])

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tool", choices=["gpt", "gemini"], required=True)
    a.add_argument("--n", type=int, default=5, help="유형마다 장수")
    a.add_argument("--model", default=None)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--dry", action="store_true")
    a.add_argument("--sleep", type=float, default=2.0)
    g = a.parse_args()
    model = g.model or ("gpt-image-1" if g.tool == "gpt" else "gemini-2.5-flash-image")
    name = "GPT" if g.tool == "gpt" else "Gemini"; fn = run_gpt if g.tool == "gpt" else run_gemini
    keyvar = "OPENAI_API_KEY" if g.tool == "gpt" else "GEMINI_API_KEY"
    if not g.dry and not os.environ.get(keyvar):
        sys.exit(f"환경변수 {keyvar}가 없습니다. 본인이 직접 설정한 뒤 다시 실행하세요.")
    log = ROOT / "기록.csv"; jobs = make_jobs(g.n, g.tool, g.seed); ok = fail = skip = 0
    for typ, src, prompt, out in jobs:
        dst = ROOT / f"결과_{name}" / typ / out
        if dst.exists(): skip += 1; continue
        if g.dry: print(typ, src.name if src else "-", out, "|", prompt[:90]); continue
        for t in range(3):
            try:
                data = fn(prompt, src, model); break
            except Exception as e:
                print(f"  재시도 {t+1}/3 {out}: {str(e)[:160]}", flush=True); time.sleep(10 * (t + 1)); data = None
        if data is None: fail += 1; continue
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
        with open(log, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([name, typ, out, src.name if src else "", f"[{model}] {prompt}"])
        ok += 1; print(f"[{ok}] {typ}/{out}", flush=True); time.sleep(g.sleep)
    print(f"끝: 성공 {ok}, 실패 {fail}, 이미 있음 {skip}, 계획 {len(jobs)}")

if __name__ == "__main__":
    main()
