# 단계1 실행 전 독립 혹독비평

2026-10-06 연구실 코덱스. PLAN.md/stage1_v1.py/preparation_v1.json 및 읽기전용 원model.py/season.py/캐시provenance 검토. reviewer는학습·원본train_y/test/EL1/잠금정답·git작업을수행하지않았다. 새파일은이사전비평뿐이다.

## 사전 판정

**현재준비단계에서 학습실행을차단할코드결함은발견하지않았다.** 이는재학습출력재현PASS가아니다. 실행중첫fold0의R3세시드와PFN문맥1에서원캐시출력가드를모두통과하기전에는수정학습집합의fit에진입하지않는순서가구현되어있다. 가드초과시허용오차를사후완화해진행하지않고실패기록을보존해야한다.

준비자료가확인하는사항:

- 원fold0/1/8전체학습·검증frame의row_id/공개sub_ec/38FULL순서hash가원캐시와일치한다. 임의raw입력과원MASK특징의불일치는이전수hash가드로통제된다.
- 버전6종·checkpointSHA·train_XSHA·공개OOFSHA·원source/seasonSHA가준비기록에고정되어있다.
-4target일과각±1일을배제한학습집합은259/268/254일,공통교집합193일이다. target훈련ID중복0 및동온실기록일차최소간격2assert가있다. 모든PFN문맥2000행을충족한다.
-4조건의query96행hash는모두32b5432df1f6dfb70f1312c6b043b31435db67d747ad56a6b987393bd95cda82로동일하다. season은학습집합마다재산정하지만이번목표4일은모두day<179이고준비결과실제query특징이동일하다. 따라서이번목표행에는'학습집합을바꿔query계절값까지달라진것'이라는우려가해결됐다.
- source를import해features/vectors/mapping/model생성자를사용하며원source의load/main을호출하지않는다. 새train_y/test읽기없음이라는범위와코드가일치한다.400일원입력weather벡터는읽지만mapping의학습기준은trainIDs로선택된날에서만fit하고queryweather를참조하지않는다.

## 실행·해석의 중요 제한

1. **replay검증범위는fold0의목표heldout2일48시간행이다.** 원전체fold0query의featurehash는확인하지만새학습출력은그중F47_160/F13_112행만대조한다. 'fold0출력전체재현'또는다른fold모든PFN문맥출력재현이라고확대하지않는다. R3기준1e-8/PFN1e-5와context_index동일성을고정한가드는긍정적이다.

2. **학습집합변경은여러학습절차를같이바꾼다.** medianimputer/MLP분할·스케일/PFN의무작위2000행문맥/타깃clip범위/훈련season변환도recipe에따라재산정된다.4queryframe이같아도훈련효과를ET/LGB/PFN문맥/학습제어변수별로분해완료한것은아니다.4조건의성공/실패점수변화는'동일입력query에같은recipe와다른훈련집합을적용'한진단으로보고한다.

3. **원예측과purged학습결과의비교대상을명확히한다.** 각원fold는다른target일을학습에포함할수있다. replay는실제heldout두날만대조하도록구현되어안전하다. 새4조건은모든4날및±1을뺐으므로같은날끼리합법비교할수있지만원fold의훈련일예측을OOF원기준으로섞으면안된다.

4. **ET첫분기차이는정확한모델내연산설명이지입력의물리인과효과가아니다.** 코드에서tree_trace는600개tree 각각의0시두query의최초분기차이/예측차를저장하며1개tree만추적하는것이아니다. tree간편향/feature중복분기를전체임포턴스처럼합치지않는다. 최종앙상블전체원인/6~23시원인/PFN실제문맥까지입증한것도아니다. imputer가열을삭제하면M.FULL[c]가오표기될수있으므로실행시transform출력열수38과get_feature_names_out순서를확인하는것이좋다. 현재원framehash만으로imputer출력열삭제여부를검증한것은아니다.

5. **캐시재개는완료NPZ+JSON쌍만안전하다.** source/preparation/train/queryhash와NPZsha를대조해일치한완료캐시를재사용하는구조는타당하다. 다만os.replace후JSON저장전프로세스가중단되면NPZ만남아다음실행assert가중단된다. traceJSON만남은상태도동일source내재실행내용이같아야한다. 이를무작정삭제/덮어쓰기/서명추정으로복구하지말고미완료상태를보존해새버전복구를등록한다. 동시에같은stage1작업을두번실행하지않는다(파일잠금은구현되지않음).

6. **학습횟수표기는group와실제모델fit을구분한다.** receipt의R3_fits12는수정4조건×3seed의R3묶음12개이며각묶음ET/LGB/MLP3회fit이다. replay까지포함하면R3묶음15개=고전모델45fit,PFN4조건×4문맥16+replay1=17fit이다. 준비manifestfit0은준비시점의상태이며실행후에도전체실험fit0으로제시하면틀리다.

7. **진단4날은채택검증이아니다.** 결과를보고gate/보정방향을설정하는후속단계는별도등록이필요하다. 부분nested12OOF스냅샷은전체DIAG/A/B및보호성능을대체하지않으며불합격안의계수추가탐색이나p기준완화로확증통과를만들지않는다. 최신EC14개선으로이번원계절v2진단을자동확대하지않는다.

## 실행 후 요구 증거

replay_verification_v1.json의실측gap/context일치,각새cache의완료쌍SHA/학습·queryID,4target시간행의raw구성원→혼합→현재누적평활→clip→seed평균재현,성공·실패/과대·과소의조건별변화를확인한뒤해석한다. 지금은준비가드PASS만확인됐고재학습출력PASS나해결성과는미확인이다.
