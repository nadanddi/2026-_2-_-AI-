#!/bin/bash
# 서버: Qwen-Image-Edit용 별도 환경(nadan_qwen) + 가중치 받기 (기존 nadan 환경은 건드리지 않음)
source ~/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q "^nadan_qwen " || conda create -y -q -n nadan_qwen python=3.10
conda activate nadan_qwen
python -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q "diffusers>=0.35" "transformers>=4.51" accelerate safetensors sentencepiece protobuf pillow numpy opencv-python-headless huggingface_hub
python -c "import diffusers, transformers, torch; print('versions', diffusers.__version__, transformers.__version__, torch.__version__, torch.cuda.is_available()); from diffusers import QwenImageEditPipeline; print('pipeline ok')"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen-Image-Edit", local_dir="/home/adilab/storage/nadan/farmai_img/data/local/weights/qwen-image-edit")
print("downloaded", p)
PY
du -sh /home/adilab/storage/nadan/farmai_img/data/local/weights/qwen-image-edit
echo QWEN_SETUP_DONE
