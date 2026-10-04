# Tweedie 정확 잎 C++ overlay 설계 점검 v1 · 2026-10-04

현재 결론은 **소스 구조상 기존 bagging .8을 보존하는 C++ renewal 설계가 가능하다**는 것이다. 빌드·설치·DLL 수정·C++ 구현·custom objective 실행·학습·예측·채점은 0이다. 설치된 binary와의 재현이 아직 없어 실행 가능한 검증완료 구현으로 부르지 않는다. 기존 family21 기각/원 환경/원 소스/캐시는 보존한다.

## 기존 in-bag를 그대로 받는 경로

공식4.7.0의 BaggingSampleStrategy는 선택된 used_indices를 SerialTreeLearner의 data_partition에 전달한다. .8/freq1의 average_bag_rate=.8>.5이므로 해당 CPU 조건에서 subset path는 사용하지 않는다. 표본 수는 각행확률.8의실현수이며 정확히 n×.8 고정 개수라고 가정하지 않는다. [bagging](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/boosting/bagging.hpp#L98), [SetBaggingData](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/treelearner/serial_tree_learner.h#L79).

RenewTreeOutput는 GetIndexOnLeaf로 현재 tree leaf의 선택행만 제공한다. non-subset의 j=index_mapper[k]는 originaltrain index; subset인 경우 j=bagging_mapper[index_mapper[k]]다. callbacks에오는 row index를 다시 train 전체 leaf예측으로 대체하지 않는다. 기존L1 renewal의 index mapping 처리와 동일해야 한다. [renew membership](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/treelearner/serial_tree_learner.cpp#L874).

GBDT는 renewal을 tree학습 후, learning-rate shrinkage와 UpdateScore **이전**에 호출한다. residual_getter(label,j)=double(label[j])−score[j]이므로 F_j=double(label[j])−residual_getter(label,j)로 기존 score를 복원할 수 있다. label_t·소거에따른double rounding은 실제감사대상이다. 첫 score는이미init 평균을포함해야 하며 첫 tree에나중에붙는 bias를또더하면안된다. [호출순서](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/boosting/gbdt.cpp#L384).

## 우선 C++ 설계: 동일 DLL의 두 모드

own overlay checkout만 허용하고 기존설치 DLL은 교체하지 않는다. 고정4.7.0 소스 commit·compiler·빌드flags·OpenMP·binarySHA를 기록한 **같은 compiled library**에서 `tweedie_exact_leaf=false` Newtoncontrol과 `true` exactcandidate를 비교한다. 이 모드는 신규 config field 기본false를 제안하며 기존objective 이름tweedie·기본metric·초기점·GetGradients·sampling·feature fraction·split gain은 유지한다. 설정parser/alias/직렬화관련생성파일은필요하지만 여기서는수정/생성하지않는다. [config 경로](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/io/config.cpp#L138).

RegressionTweedieLoss에 exactmode/λ2를보관하고 IsRenewTreeOutput는exact일때만true, false일때기존false를반환한다. Newton모드는기존leaf를바꾸지않는다. exact일때 RenewTreeOutput의 동일inbagleaf에서다음의개념계산을수행한다.

```text
check CPU+serial+one_machine+rho=1.5+lambda_l2=1
check L1=0, max_delta_step=0, path_smooth=0, no monotone constraints
A = 0; B = 0
for k in original leaf membership order:
    j = index_mapper[k] if bagging_mapper is null else bagging_mapper[index_mapper[k]]
    y = double(label_[j])
    F = y - residual_getter(label_, j)
    w = 1 if weights_ is null else weights_[j]
    A += w*y*exp(-F/2)
    B += w*exp(F/2)
find unique root of -A*exp(-t/2)+B*exp(t/2)+1*t=0
return t  # unshrunk; GBDT applies .03 once and updates all cached scores
```

실제 구현 전 finite/A/B/bracket/잔차/수렴 검사를 코드와 한국어 기록에 고정하고, 실패시 Newton fallback 없이 중단한다. solver의 고정 반복·허용오차·overflow 처리는 사전등록한다.

exact모드의training말고prediction만load할때도exp출력을보존해야한다. 계속학습까지지원하려면rho/mode/λ를model직렬화하고load생성자에서복원해야하며Newton모드의기존직렬화는바꾸지않는다. 새로운objective명은factory/metric handling까지바꿔야하므로불필요한alias변경을피하는모드설계가우선이다. [factory](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/objective/objective_function.cpp#L106).

반드시후속gradient는exact갱신후F에서재계산되어야한다. 따라서iteration2부터의tree구조는control과달라질수있으며, 이는잎갱신에따른sequential결과다. 모든tree를control구조에고정하는사후recalibration으로바꾸지않는다. sample RNG/rowmask/featurefraction변경은허용하지않고iteration별inbagIDSHA가동일한지확인한다.

## Python callback의 위험과 own-F 대안

set_leaf_output는Python에서CAPI를거쳐GBDT::SetLeafValue에도달하며model leaf값만수정한다. 해당함수는train_score_updater나valid_score_updater를갱신하지않는다. 따라서after_iteration에leaf를바꾸고nativeTweedie를계속학습하면cachedscore와modelpred가달라져원하는sequentialobjective가아니다. [Python 함수](https://github.com/microsoft/LightGBM/blob/v4.7.0/python-package/lightgbm/basic.py#L4817), [CAPI](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/c_api.cpp#L2643), [SetLeafValue](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/boosting/gbdt.h#L423).

수학상unshrunk t=−G/(H+λ), H=(A+B)/2이면 A=H+.5t(H+λ), B=H−.5t(H+λ)로inbag집계를복원할수있다. 그러나dump leaf_weight가정확H인지, epsilon빼기, gradient/hessian score_t(float) rounding, 첫treeinit bias와 .03을분리하는검사가필요하다. 공식Tree의leaf_weight는split에서passedHessian으로저장되므로실수정확A/B를그대로dump했다는뜻은아니다. [Tree 생성/weight](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/io/tree.cpp#L62).

own-F customobjective가nativecachedscore입력을무시하고own rawF로g/h를반환하며callback이실제새tree raw예측을반영한다면이론상sequential대안을만들수있다. 기존 .8 sampling은nativelearner가유지한다. 그러나dump로복원한roundedA/B는위C++의originaly/doubleF exactA/B와다른수치variant일수있다. Python alltrain합계로보정하지않는다.

own-F control은rawF0=log(originaltrain mean)을동일label_t정밀도로초기화하고, customobjective initial score 및model최종F0복원을동일하게처리해야한다. nativecustomobjective는nativeTweedie의BoostFromAverage/exp변환을자동상속하지않는다. callback후모든train행의새F와model raw전체예측+F0를비교하고, 다음gradient가이F를사용하는지검사한다. cachedtrain/validmetric은stale일수있어판정에쓰지않는다. 후속tree를nativeNewton trajectory로학습한뒤사후수정한것을대안으로슬쩍치환하지않는다. 이경로역시control재현전사용하지않는다. 단일변경의해석이명확한C++renewal이우선이다.

## 실행 전에 필요한 재현 관문

1. 원binary/compiledNewtonmode에서동일원BASE14/train/query/publictargets/±1purge/시드7로원raw_lgb를고정1e−12범위내재현한다. 실패하면solver나sampling을바꾸거나허용오차를넓히지않고stop. 이검사는미실행이다.
2. 같은compiledNewton모드 fresh/order/single/prefix재현과원binary전체66cache재현을확인한다. compiledlibrary차이로원control를재현못하면candidate전체학습금지.
3. exact첫iteration에서Newton과같은inbagID/분할·featuremask를확인하고A/B를원행loop로독립대조한다. firstbias·solverroot·unshrunk/shrunk .03·cachedscore/modelraw 일치를확인한다. renewal은그iteration의split후에하므로첫tree분할은같아야한다.
4. 이후각iteration원leafmembership/inbag만사용하며updatedscore→gradient연결을확인한다. 같은DLL모드간sampleIDSHA를비교한다. θ변경에따른later split변화와sampling변화를구분한다.
5. candidate 입력은원BASE14인지운영23인지별도사전고정해야한다. family21기각후정당한원actual계절v2 단일대조라면원BASE14를유지하고leaf만바꾸는것이scope가명확하다. 기존ET/MLP/PFN과raw혼합·shrink·clip을유지한다. 신규family/문턱은root가사전등록해야한다.

## 도구 목록과 현재 한계

읽기전용Get-Command에서git만발견했고cmake/cl/msbuild/ninja는PATH에없었다. 표준위치 `C:/Program Files/CMake/bin/cmake.exe`, `C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe`, `C:/Program Files/Microsoft Visual Studio/2022/BuildTools`, `C:/Program Files/Microsoft Visual Studio/18`의Test-Path는모두false였다. 다른디스크/portable도구전체는검색하지않아미설치단정은하지않는다. 설치/빌드승인·실행은이조사범위밖이며요청하지않았다.

가능성판정은소스경로확인이다. localDLLbuildprovenance·sameDLL원binary재현·실제sampling/solver/score연결감사는미완료다. 실험개선근거0,새학습0.
