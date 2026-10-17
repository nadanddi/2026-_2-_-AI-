# N2 대상 누적 prior 최종 독립 비평

2026-10-10. critic 실제별도무학습전수12fold검산exit0. 공식sub_temp만usecols로읽었다. target/helper/priormodel/등록SHA,NPZ/parent/metadata,원패키지보존을검사했고새fit0/EC수치0/외부0.

판정: **대상N2 구성원screen실패와전체W40GS미진입은사전계약에맞다.** DIAGRMSE .742598278→.742240903(−.048125%)이나p_worse .43585>.00625/CI상단+.00396959℃. EXT10 .889501919→.891420260(+.215665%),EXT12 .914290964→.917966303(+.401988%)로방향도실패한다. 낮은DIAG평균값만선택해전체개선을주장하지않는다.

별도전수검산:12개N2 NPZ/queryID순서·유일성·finite/shape,각source/recipe/NPZ/parentSHA,원baseline예측raw bit동일,weights/seasonnotesexact,90열×trainrows featureMASK셀receipt,train同farm검증day±1일purge를확인했다. 6RMSE는표준math.fsum계산에서root와같다. priorjoblib와target등록에내장된modelSHA 및helperactualsourceSHA도같다. 첫baseline재fit프로브는receipt0+storedparent동일성이고critic별도모델학습이아니다.

root verify_target_v1의86검사/max1.11e-16/20kbootstrap·CI·farm전후반집계는실행receipt와구현검토로확인했다. critic이이번에20k를별도재실행한것은아니다. metadatafeatureMASK는rootassert통과receipt검토이며critic독립feature재생성검사가아니다. 공식data/패키지보존SHA는직접재확인했다. 6,214,080셀은반복feature비교합으로독립표본수/전체PFN8규정실증과다르다.

실험범위: LOFO48fit는타농장단계,prior1fit는48타농장275616쌍전체의고정모델,대상13pipelinefit는baselineprobe1+candidate12이다. prior는F13/F47/F32정답없이fit되어targetfold마다같은모델cache를쓴다. 누적특징은source계약대로h0zero/첫invalidstepNaN이후day끝까지전파하며공식현재·과거입력만사용한다. preprocessing값/상호작용이타농장분포를반영하지만그출처가대상표본수준을보장하는것은아니다.

**이번결과는실제LOFOPASS가대상효용으로이어지지않은반례**다. 타농장한시간Δ형태적합성과대상잔차모델의추가효용은서로다르다. domain shift·누적drift·기존CODEX설명정보중복·다른학습분포가가능하지만원인별분해실험은하지않았다. 따라서특정원인확정/타농장전체전이불가능/시간당물리무효/정보소진을주장하지않는다. 농장·온도게이트·clip·누적계수를결과후추가하면새후보이며현재screen실패를뒤집을수없다.

원9온도모델교체/정확재현문제수정/리더보드.5127보다낮은성능달성은이번작업에서없다. 전체W40GS효과부재를입증한것도아니다. 실패기록과근거를연결해새기전설계를탐색할수있으나지금성과는보완된검증과고정후보의효용미확보다. 원제출유지/채택0/새제출0 해석은근거와일치한다.
