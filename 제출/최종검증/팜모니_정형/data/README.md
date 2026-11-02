# data — 외부 자료 없음

- 최종 답안은 **대회 제공 정형데이터만** 씁니다. 외부 자료·재배포 제한 자료는 없습니다.
- 실행할 때는 배포 원본 `train_X.csv`, `train_y.csv`, `test_X.csv`, `sample_submission.csv`를 한 폴더에 두고, 그 폴더를 실행 인자로 줍니다(README 0절).
  - 이 폴더에 넣어 두고 `data`를 인자로 줘도 됩니다.
- `code/retrain/full_retrain.py`는 EC 구성원 재학습용으로 원본 `train_X`·`train_y`에서 F13·F47 줄만 그대로 옮긴 사본을 작업 폴더에 만듭니다(줄 내용 변경 없음).
- 2026-10-17 재현에 쓴 원본 파일의 SHA256:
  - `train_X.csv` 15,278,580 B `21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300`
  - `train_y.csv` 5,534,334 B `347a953051b3220d4e7e144875da0cdf289a8ce972e4ed0929e2a93f18d006e9`
  - `test_X.csv` 118,141 B `8e6bae5f81356470c5e63def9569e3d7a1be15b9574a4b830c405f1183e6cdad`
  - `sample_submission.csv` 34,584 B `a5a6777a6ceab18eb6a61215ae1399eb35b22b62622c3c3a0469f39df84c109f`
