#!/bin/bash
# 서버: SDXL 계열 가중치 (가짜 제작 5판용)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img; mkdir -p local/weights
python - <<'PY'
from huggingface_hub import snapshot_download
pat = ["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*", "scheduler/*"]
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", local_dir="local/weights/stable-diffusion-xl-base-1.0", allow_patterns=pat)
snapshot_download("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", local_dir="local/weights/sdxl-inpainting", allow_patterns=pat)
snapshot_download("madebyollin/sdxl-vae-fp16-fix", local_dir="local/weights/sdxl-vae-fp16-fix", allow_patterns=["*.json", "*.safetensors"])
print("XL_DONE")
PY
