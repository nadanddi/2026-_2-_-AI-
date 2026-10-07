# BLK baseline 구현 소스 조사 v1 — 2026-10-07

읽기 전용 조사이며 학습·GPU 실행·다른 폴더 수정은 없다. 아래 경로는 저장소 루트 기준이다. 동작을 가져올 때 main/top-level import 대신 순수 함수만 추출하거나 새 own 모듈에 명시적으로 구현해야 한다. WT1/WT2는 top-level에 GPU import·외부 prepare·CSV 읽기·채점 동작이 있으므로 통째로 import하지 않는다.

## 배포14와 최신0.4 기준선 구분

- `집/클로드/submission14_ec_sg2/model.py`: 배포 R3 seed7/101/2024, PFN context1/2/3/4, PFN0.2. `features`, `run_len`, `ops_day`, `shrink`, `et`, `lg`, `mlp`, `predict_all`이 핵심이다. 파일은 `season.identify/vectors/mapping/W`, `sg2post`에 의존한다.
- `집/클로드/research/ec3_WT1_dp1_removal_pfn_share_v1.py`: 최신 채택0.4 근거용 R3 seed47/1414/6464, PFN context5/6/7/8, GPU 출력. 이 파일의 `pfn`은 CUDA이므로 그대로 호출 금지. source의 CPU equivalent를 별도 pin해야 한다.
- `집/클로드/research/ec3_WT2_pfn_share_with_sg2_v1.py`: WT1 저장 shrunk members에서0.6R3+0.4PFN→clip→SG2→clip한 sg4 참조. `correct`는 외부 global WV/hrs/SIG/alld/ec/lockd/PV/roles 등에 의존하므로 순수 함수가 아니다. 입력 인자로 전부 분리해야 한다.

## 모델·특징·순서

ET: median imputer→600trees/max_features1/min_samples_leaf1/random_state seed. LGB:800trees/lr.03/leaves31/min_child40/subsample.8+freq1/colsample.8/reg_lambda1/Tweedie power1.5/deterministic force_col_wise. MLP: median imputer→StandardScaler→(128,64)/alpha.01/lr.001/max_iter800/early_stopping/validation_fraction.12/n_iter_no_change25. CPU threadpool은사전등록값으로고정하며seed포함get_params를저장한다.

BASE14는원입력10+hr_sin/hr_cos/midnight+season. FULL38은BASE14+10개h0+7구동기누적평균/0비율14개. R3에DP1 9개를추가하여ET47, LGB/MLP23개. PFN은FULL38이며DP1없음. 원외기4는season/SG2에쓰이지만본체직접열이아니다.

R3 raw=.6ET+.3LGB+.1MLP. 최신연구baseline은각seed에대해 .6R3+.4공통PFN평균, 하루0..h prefix shrink(.5현재+.5누적평균), trainEC min/max clip, SG2, clip 순서다. WT1 저장ET/LGB/MLP/PFN은이미shrink되어있으므로저장member를재shrink하면다른모델이다. 선형shrink는혼합과교환가능하나clip/SG2는그렇지않다. 새BLK에서는각raw단계출력부터저장해중복shrink/clip을방지한다.

PFN CPU adapter: `TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', model_path=봉인checkpoint, n_estimators=4, random_state=context_seed, ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)`가배포구현이다. trainFULL38을float32, y는float로주고`default_rng(context_seed).choice(len(tr), size=min(2000,len(tr)), replace=False)`로문맥을고른다. 최신baseline은contexts5..8로고정, sampling전train orderedIDs를봉인한다. checkpoint실제경로/가중치SHA와tabpfn/torch버전은추가확인필수이며조사만으로CPU/GPU수치동등성이입증되지않는다. full batch vs단일query·순서불변성검사를최종PFN출력에서한다.

## SG2 이식의 핵심 P1

`집/클로드/submission14_ec_sg2/sg2post.py`의 `_ident`, `prepare`, `ref_calendar`, `correct`가배포SG2핵심이다. `prepare(x_all)`는pass1 WV 표준화mu/sd를x_all의pass1전체에서계산한다. **BLK의pass1 query를train과합쳐넘기면query 통계로전처리를fit하여누수한다.** 실제test는pass2여서이결함이표면화하지않을수있지만BLK에서는반드시분리해야한다.

개선API는 `fit_reference(reference_inputs,reference_labels)`로WV표준화·refcalendar·시간별SIG표준화·공개정답일평균을고정하고, `predict_query(prefix, base_prefix_predictions, frozen_reference, apply_scope)`로현재prefix를transform/검색하도록한다. gap/hidden labels는loader에서없어야한다. 과거query의완료하루를참조하는것은현재보다앞일때만허용하며뒤query의하루전체pivot/role/달력연결은금지한다. SG2의greedy second역할을24h동일외기로추정하는경로도source timing을감사한다.

SG2조건: h>=5의외기prefix twin(zRMSE<=.05), 실패시이전record full_date+.1,참조cal거리<=3·같은cal제외,13개prefix제어서명거리+.15날짜거리,nearest일평균EC=a1,현재prefix예측평균pm, |a1-pm|<=.30이면현재예측+.5(a1-pm). 서명mean/sd는ref-only; empty/zero-sd/nofinite-distance/firstrecord fallback명세필수. gap때문에직전record가없는경우 days index-1이뒤record로감싸지않도록확인한다.

배포`correct`는day<179 skip,WT2`correct`는TM전체에적용한다. BLK_RAW_PASS/BLK_QUERY_ROLE 별도사전명세없이동일baseline으로부르지않는다. 원numeric day는변경하지않고apply_scope flag만별도등록한다.

## fit 열기 전 요구

실제CSV와모든의존SHA검사,train/query/anchor/calendar/PFNcontext orderedID계약,raw→혼합→shrink→clip→SG2단계검산,최종future/otherfarm/order/singlequery불변성,CPU허용오차1e-6 gate를통과한다. BLK가pass1뿐이라는한계와전체baseline미재현상태를보고한다. 이조사자체는모델채택이나성능검증이아니다.
