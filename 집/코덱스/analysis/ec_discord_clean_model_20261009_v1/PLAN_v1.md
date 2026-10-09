# 고EC 불일치일 통삭제 데이터 + EC 모델 — 사전 계획 v1
2026-10-09 집 코덱스, 사용자 ‘데이터를 통으로 지우고 데이터를 새로 만든 다음 EC예측 모델 만들어봐’.

## 목표/고정 범위
원본 보존, 같은5NN proxy로 설명하기 어려운 고EC일24시간 전체를 제거한 새X/Y데이터를 만들고 모든 구성 요소를 그 자료로 학습한 standalone EC후보모델을 저장한다. 원본 행번호는 유지하며 재번호 부여/값 수정/재귀 삭제 없음. 이상치나 라벨오류 확정 아님.

CV용 입력은 기존 공개360일/8640행·동결된DIAG10/±1purge/seed7,101,2024 그대로. train_y 원본의 소비된lock40일은 CV가 모두 완료되고 결과가 저장된 뒤 최종학습자료400일 제작/모델fit에만 포함. lock40재채점·튜닝·검증0. 실제test 입력/예측/점수로 선택0, 이전대회 자료 미사용.

## 선별과 데이터 처리
규칙 고정: 하루EC평균>=1 & 같은farm의 자기·±1기록 제외47개FULL_R3 일평균입력 거리5NN의 EC중앙값보다 >.5. median대치/mean/std/계절변환은선정기준outertrain으로fit. CV 학습삭제는오직outertrainlabels로결정. q제거는outertrain reference와q평균입력/실제일평균EC로판정하는 조건부채점용이며q의label은학습삭제/전처리/모델로유입0. query전체일입력은선별채점에만; 모델은현재/허용prefix RAW10과고정된계절보간만.
전체public360용cleanCSV/제거표는데이터정합성 및 별도자료로저장하지만CV학습에global삭제목록을사용하지않는다. fold별cleanX/Y/제거훈련·검증일표 및selector행렬/이웃/삭제IDs를저장. 삭제후cleanoutertrain만으로season/각imputer/scaler/clipbounds/모든모델재fit. 삭제된자료는모델학습의어떤경로에도없음. 선별기 자체는삭제전자료를1회읽는정제단계.

## 모델과 공정 비교
새모델R3=.6ET+.3LGB(Tweedie)+.1MLP,3seed평균, 기존 FULL_R347/BASE_R3설정·ET600/LGB800/MLP128,64·.5현재+.5허용prefix평균 후trainbounds clip. PFN/SG2는구성에포함하지않는별도R3모델이다(기존EC14와동일구성이라고말하지않음).
원자료동일R3 baseline은동결된ET/LGB/MLP raw cache에서동일비중/후처리 재구성; source/feature/train/query/predhash확인 후 재사용. fold0seed7 ET·LGB·MLP를새factory로재현후(max1e-10, MLP허용1e-8)candidate시작. baseline constant=train ECmean 저장. 동일q·동일filteredq로비교. CV신규후보90모델(각30)·baseline재현3·최종9. 성능나빠도사용자가요청한데이터/후보모델은끝까지제작,채택0/제출준비0.

## 사전 점수/판정
공동필수: 정제조건부q와원q전체를각각비교. 조건부표본에는qlabels로선정된다는위협을매번명시하며실제평가성능개선으로주장하지않음. 전체/조건부/제거q/일반/고EC/농장×구간/시드/fold/학습RMSE·foldSD·bias를저장.
단일candidate에대한두공동비교(조건부/전체) 농장층화day//5 상대기록번호구간bootstrap20000, RNG20261009, .025/2=.0125 기준 및모든3seed같은개선방향. 원q전체또는pass2전체≥2%악화면효용실패. A/B/독립최종holdout/새seed검증없으므로 어떤결과도 공식채택/실력확증 아님. 전체모집단과조건부모집단의스코어를직접비교해삭제이득을계산하지않음.

## 산출물/검증
ownanalysis에계획/스크립트/등록/프로파일/결과/3단계비평,ownlocal에public·fold·full최종cleanCSV/제거목록·학습9joblib/season·bounds·feature/modelmanifest. 제출형식CSV/재현ZIP/제출확정구성 생성0.
최종400일train_y는CV완료hash있을때만load/정제/fit, 성능채점40일없음. 데이터프로파일(shape/dtype/missing/key/hour/음수/중복/분위/coverage)전후와cleaninglog 필수. X/Y동일ID·24시간통삭제·원본행보존·유한target·원본/결과hash.
모델저장reload 실제예측일치,예측용훈련단일표본순서/시점이후·타farm입력변조에허용행예측불변,queryweather사용0,부트스트랩/핵심3숫자 독립재계산. finalfit과CV예측 artifact분리. 실행환경/버전/seed 및지원codehash저장,전체run단계스크립트로재현가능.

## 독립 비평
계획/code→첫fold→전체score/최종모델 3단계. 조건부모집단선택/원자료정보잔류/전처리refit/멀티시드/소수일집중/누수/보관모델재현을공격. 발견수정은새버전. 정상dataset선택효과와전체실전효용/라벨오류/물리원인 분리.
