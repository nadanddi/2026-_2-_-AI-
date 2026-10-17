"""Colab T4 실측용 (제출 ZIP이 524장 5분 안에 끝나는지).
사용법 (Colab, 런타임 → 런타임 유형 변경 → T4 GPU):
  1) 이 파일, 제출 ZIP(예: 팜모니_나단4.zip), test_images_521.zip 을 구글 드라이브 MyDrive/farmai_t4/ 에 올린다.
  2) 새 셀에서:
       from google.colab import drive; drive.mount('/content/drive')
       !pip -q install safetensors opencv-python-headless
       !python /content/drive/MyDrive/farmai_t4/colab_시간측정.py 팜모니_나단4.zip
  3) 마지막 줄의 '합계'가 300초보다 충분히 작은지 본다(524장 기준으로 환산해 출력).
채점과 같은 순서: ZIP 풀기 → load_model → predict(전체 이미지 목록). 네트워크 접근 없음.
"""
import sys, time, zipfile, shutil, importlib
from pathlib import Path
BASE = Path("/content/drive/MyDrive/farmai_t4"); WORK = Path("/content/t4work")

def main(zip_name):
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    t = time.time(); zipfile.ZipFile(BASE / zip_name).extractall(WORK); zipfile.ZipFile(BASE / "test_images_521.zip").extractall(WORK)
    print(f"압축 풀기 {time.time() - t:.1f}s (채점 시간에는 보통 포함되지 않음)")
    team = next(p for p in WORK.iterdir() if p.is_dir() and (p / "inference.py").exists())
    imgs = sorted(str(p) for p in (WORK / "t525").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음(CPU)", "| torch", torch.__version__)
    sys.path.insert(0, str(team)); I = importlib.import_module("inference")
    t0 = time.time(); m = I.load_model(str(team / "model")); t1 = time.time()
    pr = I.predict(m, imgs); t2 = time.time()
    n = len(imgs); per = (t2 - t1) / n
    print(f"이미지 {n}장 | 모델 로드 {t1 - t0:.1f}s | 예측 {t2 - t1:.1f}s (장당 {per:.3f}s)")
    print(f"524장 환산 합계 ≈ {t1 - t0 + per * 524:.1f}s  (한도 300s, 여유 {300 - (t1 - t0 + per * 524):.0f}s)")
    print("확률 범위", round(min(pr), 4), round(max(pr), 4), "| 0.5 이상 비율", round(sum(p >= .5 for p in pr) / n, 3))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "팜모니_나단4.zip")
