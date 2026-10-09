# 9회차 온도 W40G-S 재현 계획

사용자 직접 요청(2026-10-08)에 따라 온도 모델만 재현한다. 제출9회차/submission_14의 온도는 W40G-S, temp_candidate_v13이다. 다른 온도 개선 실험이 아닌 기존 패키지 실행 검증이다.

1. 제출/09회차_2026-10-06(팀)/팜모니_정형데이터_재현패키지_14.zip의 온도 소스/README/config/기준CSV와 루트 답안을 읽었다. 원본CSV3개는온도폴더에미포함, 공개 TabPFN체크포인트는같은ZIP의EC폴더에동봉됨.
2. 내local/temp_round9_reproduction_20261008_v1/extracted 아래 ZIP을 새로 안전추출한다. 압축파일 내 경로가 추출루트 밖으로 벗어나면 중단. 실행소스 원본 bytes 보존 및SHA등록. 공식CSV3개는해시 일치 확인 후 추출본data에복사한다. 원ZIP·제출CSV·타인폴더변경0.
3. Python3.12 + repository .analysis-tools/python(core),extra(torch/TabPFN),msvc 및ZIP DLL로런타임연결. 각패키지버전과경로기록. 온도requirements와비교. 모델체크포인트2ab5a07d…를같은ZIP에서tasklocal캐시에연결하고offline/telemetryoff. 기존캐시예측재사용0, BASE/CODEX/PFN8문맥을패키지동일설정으로새fit/predict.
4. 실행전 독립계획검토. 첫 BASE/CODEX완료시 로그/입력/출력경로 중간검토. 완료후1440행ID/순서/finite 및각온도값을제출CSV와ZIP온도기준CSV에대조. 정확6자리일치여부와maxabs/RMS/차이행수를보고한다. 원문에허용오차없으므로재현PASS를사후수치에맞추지않음: strict PASS=1440온도값정확일치, 다르면근사값과원인별도보고(실행성공과정확재현구분).
5. temp_v13_checks의전모델8회추가학습은우선재현비교에필요하지않으므로자동실행하지않는다. 기존규정검사결과와이번완전재학습/수치비교범위구분. 최종독립검산은표준CSV/math.fsum·npz혼합식/게이트재구성/입력출처·체크포인트·원소스SHA검증으로한다.
6. 신규출력은ownlocal 재현용이며새제출/채택0, 평가정답/잠금검증재채점0. 최종학습에서는패키지대로공식학습400일 사용(기존제출재현목적). 중간·최종독립검토와作業일지/HANDOFF/카탈로그결과추가까지완료한다.

실행환경만연결하는wrapper는내analysis에새로작성. 원모델소스/파라미터·문맥·시드변경금지. 실행제약이확인되면환경수정으로해결하고원예측방식변경은피한다. byte동일/숫자동일/근사구분을명시한다.
