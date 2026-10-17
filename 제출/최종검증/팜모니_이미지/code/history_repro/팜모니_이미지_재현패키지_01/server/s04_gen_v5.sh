#!/bin/bash
# 서버: 가짜 제작 5판 생성 큐 (관리자 규칙: conda activate nadan, 시작 전 nvidia-smi 확인, 빈 GPU만 사용)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data
GPU=0
echo "== $(date '+%F %T') nvidia-smi 확인 =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# 다른 사람 프로세스의 사용량만 계산(작업 폴더가 farmai_img인 것은 우리 작업)
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU); OTHER=0
while IFS=, read -r bus pid mem; do
  [ "$(echo $bus)" = "$BUS" ] || continue; pid=$(echo $pid); mem=$(echo $mem | tr -dc 0-9)
  case "$(readlink /proc/$pid/cwd 2>/dev/null)" in *farmai_img*) ;; *) OTHER=$((OTHER+mem));; esac
done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits)
echo "GPU $GPU 다른 사람 사용량 ${OTHER}MiB"
if [ "$OTHER" -gt 2000 ]; then echo "GPU $GPU 다른 사람이 사용 중 → 중단"; exit 1; fi
while ! grep -q XL_DONE ../logs/weights_xl.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8
# 1) 유형별 2장 시험 생성(오류 조기 발견)
for k in filter_blur composite_v5 ai_gen_xl full_style_xl partial_style_xl partial_gen_xl; do
  python -W ignore image/im15_fakes_v5.py $k 2 $GPU || exit 1
  n=$(ls local/im_set_v5/img/${k}_0000[01].jpg 2>/dev/null | wc -l); if [ "$n" -lt 2 ]; then echo "SMOKE_FAIL $k (생성 $n장)"; exit 1; fi
done
echo SMOKE_OK
# 2) 본 생성
for kn in "filter_blur 400" "composite_v5 900" "ai_gen_xl 900" "full_style_xl 900" "partial_style_xl 900" "partial_gen_xl 900"; do
  set -- $kn; echo "== $(date '+%F %T') $1 $2 =="; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  python -W ignore -u image/im15_fakes_v5.py $1 $2 $GPU
done
echo GEN_V5_DONE
