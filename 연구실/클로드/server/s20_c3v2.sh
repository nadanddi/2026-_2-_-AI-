#!/bin/bash
# 서버: C3v2 (IM_C3v2_v1.md) — 학습 묶음 → 특징 → 최종 학습·내보내기(나단5) → self_check → 봉인 판정
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
python -W ignore -u image/im34_make_c3v2_sets.py || echo FAIL_sets
for S in im_set_newreal im_set_v7tr im_set_qwentr; do
  M=local/$S/manifest.csv; I=local/$S/img
  python image/im06_lowlevel.py $S $I $M || echo FAIL_low_$S
  python image/im06b_region_feats.py $S $I $M || echo FAIL_reg_$S
  python -W ignore image/im04_clip_feats.py $S $I $M || echo FAIL_clip_$S
  python -W ignore image/im16_backbone_feats.py B2 $S $I $M || echo FAIL_B2_$S
  python -W ignore image/im23_patch_feats.py $S $I $M || echo FAIL_P16_$S
  python - $S <<'PY' || echo FAIL_look_$S
import sys, csv; sys.path.insert(0, "image")
import im14_lookup_feats as K
s = sys.argv[1]; rows = list(csv.DictReader(open(f"local/{s}/manifest.csv", encoding="utf-8")))
K.run(K.DB(), s, K.LOCAL / s / "img", rows, [False] * len(rows))
PY
done
P5=../submit_img05/팜모니_나단5; mkdir -p $P5/model; cp ../submit_img04/팜모니_나단4/*.py ../submit_img04/팜모니_나단4/requirements.txt $P5/
python -W ignore -u image/im20e_train_final_img05_C3v2.py $P5/model ../submit_img01/팜모니_나단1/model > ../logs/build_img05.log 2>&1 || echo FAIL_build
tail -4 ../logs/build_img05.log
SC=~/storage/nadan/공용/대회자료/이미지데이터/참가자_배포/self_check.py; python -W ignore $SC --team_dir $P5 2>&1 | grep -e RESULT -e 실행 -e 지표
python -W ignore -u image/im35_sealed_eval_c3v2.py > ../logs/sealed_c3v2.log 2>&1 || echo FAIL_eval
echo C3V2_DONE
