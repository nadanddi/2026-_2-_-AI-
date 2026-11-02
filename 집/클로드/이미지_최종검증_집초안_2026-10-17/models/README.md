# models — 저장 모델·가중치

저장 모델과 사전학습 가중치는 모두 채점에 쓴 추론 ZIP `팜모니_나단11.zip`의 `model/` 안에 있습니다. 같은 파일을 여기에 다시 넣지 않았습니다(안내문 2절: 채점 ZIP과 같으면 생략 가능).

- ZIP 전체: 916,144,728 B, SHA256 `6fe04909341d104dfe7084905f05457b862ceacbd158e645b278520f439b27c1`
- 아래는 ZIP을 푼 각 파일의 SHA256입니다(2026-10-17 계산). `code/inference_나단11/`의 코드·설정 파일은 이 표의 같은 이름 파일과 바이트 단위로 같습니다.

| 파일 | 크기(B) | SHA256 |
|---|---|---|
| `팜모니_나단11/inference.py` | 15,869 | `6f8b7c406a43da0a03d9b36b05871a7c5a307db0c0e1a7cfbb96d3c26afaf43c` |
| `팜모니_나단11/lowfeat.py` | 2,515 | `0d266bcfbbd3b849e7f31afb16da1c7015b5b4fb885846f3699118344719dd15` |
| `팜모니_나단11/vit.py` | 2,250 | `492e2990bc44d989a29a469ff2421647d481f3bba04e57fce971b998ce93d1df` |
| `팜모니_나단11/requirements.txt` | 122 | `a4f50e86bfb09674a53afca1d512520e55a66e648f502bcb8cff729d17667d99` |
| `팜모니_나단11/README.md` | 18,359 | `5bcf72cf48bb3feb63f5d06fcbd4ae42a9360ca8db1721ca806a20155533df50` |
| `팜모니_나단11/model/config.json` | 1,436 | `28ccd97ea35636edac97e97b742e93cb618d7eace34b456e31710ecccf4e0f62` |
| `팜모니_나단11/model/b2_visual_fp16.safetensors` (CLIP ViT-L/14@336 시각부, OpenAI, MIT) | 608,620,512 | `2f03df214a8d0310ac018a40fd5255119d4b97031daa95cc91d1ad4adb538a19` |
| `팜모니_나단11/model/b16_visual_fp16.safetensors` (CLIP ViT-B/16 시각부, OpenAI, MIT) | 172,401,640 | `d3b2627711293703f807734266d63db856c3afb6d1494b256f9d615a6fbdd140` |
| `팜모니_나단11/model/b2_lr.npz` | 19,434 | `81abff516f71456f5394f1e93510f29c2e933c5927fc30dc9091388a10e37e3c` |
| `팜모니_나단11/model/low_hgb.npz` | 420,752 | `c18a18dfb6e1c3c6d025d8df02bcd95cb338f064f66ebcf1087a075241fb622d` |
| `팜모니_나단11/model/e_pg_lr.npz` | 19,434 | `d4ac8ae133c024e5a4a3c3c67b5e0851b762bab3425c92ad17a17be4bf73b5f7` |
| `팜모니_나단11/model/e_ps_lr.npz` | 19,434 | `41c13a47b8d98352c8caac1fd7dbfbd5a1b0182e9bfe99649aa7688024926cfb` |
| `팜모니_나단11/model/e_l_lr.npz` | 13,290 | `fb784953bba184d70b0b84f1caaec066cd6309c2827ecbc18117368e7ec6fdb9` |
| `팜모니_나단11/model/h_experts.npz` | 25,054 | `c55be76ecb0a744da9e17c23f78c00798453244f68905709a697ab19f13d4d41` |
| `팜모니_나단11/model/gr_expert.npz` | 10,718 | `4002128664b9a70ddf03b5f429d83b421d800c3a9f4eef4aebcde45b9fd8d4e7` |
| `팜모니_나단11/model/gpt_expert.npz` | 10,670 | `d9a86907609e4332a28d17b8489414fec91a9ba7d5ce9fa83e2bcff618b12ccf` |
| `팜모니_나단11/model/fp_db.npz.xz` (AI Hub 파생 통계) | 74,398,140 | `c389d3c5a475c1f76e71b94403eae721e19ad3c4c2d0879781849e515c6589a6` |
| `팜모니_나단11/model/finefp_db.npz.xz` (AI Hub 파생 통계) | 118,959,960 | `c04584f3a0c49b6cd67648efd3fcdd79becb14d24d8bbf028324356be4aefe99` |

- CLIP 가중치 출처: OpenAI CLIP(MIT 라이선스). open_clip 3.3.0의 `ViT-L-14-336`·`ViT-B-16`(`pretrained=openai`)에서 시각부만 fp16으로 저장했습니다(`code/training_연구실클로드/im20_train_final_img01.py`, `im20e_train_final_img05_C3v2.py`).
