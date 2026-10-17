# N4 첫 fold 독립 중간 비평

2026-10-10 연구실 코덱스. critic의 `n4_critic_first_v1.py`를 별도로 실제 실행하여 정상 exit0을 확인했다. 새 모델 fit/predict 0, 온도 정답만 usecols로 읽었으며 EC 수치 정답 읽기0, 외부자료0. 결과는 `n4_critic_first_v1.json`에 새 이름으로 저장했다.

**판정: 첫 fold의 출처·분할·가중치·계절·PFN 문맥·혼합 검산이 통과했다. 같은 `n4_run_v2.py`/등록으로 전체12fold 예비실험을 계속할 수 있다.** 첫fold 소폭 개선을 전체효용이나 채택 근거로 쓰지는 않는다. 예비실험 전체가 끝난 뒤 독립 최종 검산·비평이 필요하다.

## critic 실제 확인

- 현재 원package54파일·공식3CSV·checkpointSHA가 원preparation 및source receipt와 일치했다. 현재preparation파일SHA와실행source/runtimeSHA도 source receipt·등록과 맞았다. recipe canonicalSHA와 첫NPZ파일SHA/receiptSHA 계약도 확인했다.
- orderedquery960ID·train8256ID가등록/receipt와 정확히 일치했다. 양쪽 ID중복·교집합0, 각농장·날의0..23시완전24행, 같은농장query±1기록일과train분리도 확인했다.
- 등록 train IDs의원학습입력만으로 `runtime.weights`를 다시 계산했다. .04계열23행/.2 1995행/1 6238행으로 receipt와 정확히 일치한다. 계절도등록train/query날을동일불변helper로새로계산하여 notes 전체가 receipt와같았다. 이는기존helper의새재생검사이며 독립PAV알고리즘으로수학방법을다시구현한것은아니다.
- 새가중치로 `default_rng(49).choice(...,2000,replace=False,p=w/w.sum())`를 재생했고 **PFN 문맥2000 ID의순서까지** NPZ와정확히같았다. context중복0·모두train subset,query지지유입0이다. 문맥일치가PFN예측전체인과성·다문맥효용을뜻하지는않는다.
- 원query현재공기온도에서별도scalar gate를재구성해storedgate와exact일치했다. 전체960행에서scalar `math.fsum`으로기준혼합 및후보차를계산한최대차는7.11e−15다. g=0 36행은baseline/candidate raw exact같고모든멤버/혼합/게이트는유한했다.
- query공개온도정답을후처리산술에만사용해 RMSE를독립fsum으로계산했다: **기준 .7271330403509996 → 후보 .7248781265765775**. 기존verify12검사결과와같다. 20k최종bootstrap은첫fold에실행하지않았다.
- CODEX726/9109 no-op은실제runner의매fold `array_equal` assertion 및첫receipt `seed_noop_max=0`로확인한다. critic은동일모델을재학습하지않았으므로독립재fit으로다시증명했다고쓰지않는다.

## 범위와 계속 조건

실제첫실행로그의PFN fit후DONE,부모가전달한정상exit0,원NPZ/receipt 및critic독립재생검산을연결했다. 원54파일/sourcepin 확인과4,464,900셀감사상속범위도유지한다. **PFN49한문맥·원3BASEseed평균의첫960행 확인**이지원PFN8표본·모든seed·전체12fold·full predict미래/타농장/배치불변성의증거는아니다.

다음전체실행은동일등록/source/계수/threshold/분할을유지한다. 첫cachedfold는원recipe+파일SHA검사로재사용하고뒤fold만계속한다. 모든fold에서CODEX726/9109no-op에실패하면중단하며어느부분설정을튜닝해계속하지않는다. 최종은등록한3validator방향/pooledbootstrap alpha.025/6·CI계약으로예비판정하고, 통과해도원8문맥및추가실제stochastic변동/확인layout 전에채택0이다.
