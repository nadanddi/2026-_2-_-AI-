#!/bin/bash
# 채점 환경(안내 7절) 버전 재현용 새 환경 nadan_t4 — 관리자 규칙: conda create, python -m pip install
source ~/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q "^nadan_t4 " || conda create -n nadan_t4 python=3.13 -y
conda activate nadan_t4
python -m pip install torch==2.11.0 torchvision==0.26.0 numpy==2.1.3 pillow==11.3.0 opencv-python==5.0.0.93 safetensors==0.8.0 scikit-learn==1.6.1 pandas==2.2.3 scipy==1.16.3
python -c "import sys,torch,numpy,cv2,PIL;print(sys.version.split()[0],torch.__version__,numpy.__version__,cv2.__version__,PIL.__version__)"
echo T4ENV_DONE
