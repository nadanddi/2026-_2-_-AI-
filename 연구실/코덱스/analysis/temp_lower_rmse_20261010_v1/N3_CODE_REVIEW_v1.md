# N3 코드 실행 전 독립 비평

2026-10-10. n3_model_v1/n3_run_v1 직접읽기만, criticfit0/EC0/외부0.

**차단점1: 사전bootstrapseed불일치.** N3_PLAN_v1은20261010인데run_v1 registration과default_rng는20261009이다. 현재fit0이므로새runner에서등록·실행seed를계획20261010으로일치시키고원소스보존한다. 결과후seed선택하지않는다.

이외정적model경로는계획에맞다. query.sub_temp존재를함수첫줄에서reject하고caller는그열을삭제한다. shape목표y−trainy0는tr에서만생성한다. level은trh0/w0/PHYS19+season20/Ridge100+LGB120/.03/8/12/lambda15. query0시예측을rawfarm+t//24키로모든시간에연결하며검증label을사용하지않는다.

shape는PHYS19 current−同dayh0 후h0difference0,keepempty median→ScalermeanFalse→RidgeinterceptFalse로fit하고CF89 LGB220잔차를더한다. queryh0의shapephysics예측0 및최종h0=levelpred exactassert가있다. rawshape0는모델예측을同day키로빼는것이고shape0truth가아니다. sin/cos/farmconstant도PHYS동일차분규약이다.

source/parentSHA・firstbaselineexact・훈련외MASK89열eq・원weightsseasonnotes와split키가이전검증된공통코드에연결된다. 하루24행과at0유일성검사,원feature89/19schema를보존하며NaN처리는계획대로다. all24uniquehours는원feature초기assert와parentcomplete날출처에연결하나model.predict자체에hoursetassert를추가하면더명확하다.

남은필수negative증거: `n3predict(..., va.sub_temp가있는frame,...)`를소형무학습호출해첫줄assert로차단됐음을기록한다. 현재댓글의'labels를추가해도prediction영향없음'은실제기능이예측동일성이아니라**학습전에거부**이므로표현을정정한다. 이guard검사는querylabel임의변조후같은모델예측bit불변검사를실행했다는뜻이아니다.

fit수설명: `newfits+=2`는N3날수준/shape두head파이프라인이며각headRidge+LGB를포함한다. total25는12candidate×2heads+baselineprobe1,새candidate모델은12twohead조건이다. 내부estimatorfit개수와'CODEX pipelinefit25'라는표현은구분해야한다.

NPZ N3/level/shape_raw/shape 저장은후속raw0차감·최종0시·MSE항등식/cov검산에충분하다. ROOTfeature감사와source정적확인은전체모델교란예측실증이아니다. bootstrap정합성수정및negativeguard기록후첫fold계속가능하며아직효과판정하지않는다.
