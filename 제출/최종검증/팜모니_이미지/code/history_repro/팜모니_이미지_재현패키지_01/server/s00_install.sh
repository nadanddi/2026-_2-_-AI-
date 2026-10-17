#!/bin/bash
# 서버(ADILAB) nadan 환경 패키지 설치 — 관리자 규칙: conda activate nadan, python -m pip install 만 사용
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nadan
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install open_clip_torch diffusers==0.32.2 transformers==4.46.3 accelerate safetensors scikit-learn pandas pillow opencv-python-headless scipy
python -c "import torch,open_clip,diffusers;print(torch.__version__,torch.cuda.is_available(),open_clip.__version__,diffusers.__version__)"
echo INSTALL_DONE
