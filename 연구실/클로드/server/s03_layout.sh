#!/bin/bash
# 서버: 로컬과 같은 경로 구조 만들기 (스크립트는 data/image/*.py, LOCAL = data/local)
cd ~/storage/nadan/farmai_img/data
[ -e local ] || ln -s . local
[ -e weights ] || ln -s ../local/weights weights
mkdir -p ~/storage/nadan/공용/대회자료/이미지데이터/참가자_배포
ls -la | head; ls local/ | head
