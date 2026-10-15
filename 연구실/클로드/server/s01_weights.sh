#!/bin/bash
# 서버: 모델 가중치 직접 다운로드 (HF 허브) — 로컬과 같은 경로 구조(local/weights/...)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img; mkdir -p local/weights
python - <<'PY'
from huggingface_hub import snapshot_download
import open_clip
for rid in ["stable-diffusion-v1-5/stable-diffusion-v1-5","stable-diffusion-v1-5/stable-diffusion-inpainting"]:
    snapshot_download(rid, local_dir=f"local/weights/{rid.split('/')[1]}", allow_patterns=["*.json","*.txt","*fp16.safetensors","tokenizer/*","scheduler/*"], ignore_patterns=["*.ckpt","*non_ema*","*.bin","*ema*"])
open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", cache_dir="local/weights/openclip")
print("WEIGHTS_DONE")
PY
