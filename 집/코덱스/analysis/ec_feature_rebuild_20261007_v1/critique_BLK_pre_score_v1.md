# BLK 채점 전 독립 비평 v1 — 2026-10-07

verify_BLK_baseline_v1.py, blk_score_v1.py/v2.py 및실제BLK_ASSEMBLED_v2의lineage/저장출력을읽기전용검토했다. 보류EC정답을읽거나채점하지않았다. 현재assemblygate=False를확인했다. broadPFN7/8감사통과전에verify/scorer를실행하지않는조건을유지한다.

## 실제 산출물 확인

assembledpredictionSHA가manifest와일치하고member파일/assembler의존SHA도현재파일과전부맞는다. baseline6개/candidate18개·seed47/1414/6464단계출력을확인했다. guard활성0이며각scope/seed의GUARD출력은baseline과정확히같다. 이것은기존mutualgraph규칙의fallback결과이지CH2기작전체시험결과가아니다.

## gate/수식/분모 검토

verifier는9R3future/4PFNcontext감사·actualsource/weight/registration/feature/context계약·288postprocess감사를확인한뒤1440행finite/ID/guard동일성을검사한다. rawmember혼합·oneprefixshrink·clip을독립stdlib로재계산하고RAW_PASS출력과1e-12이내일치를요구한다. Query-role는SG2참조정책adapter감사와실제postprocess감사로연결한다. CPUrecipebaseline임을명시하며historicalGPU/0.2배포동등성주장을하지않는다.

scorer는verifiedreceipt와prediction/rulesSHA를먼저확인하고그뒤에만1440querytruth를읽는다. 전체행을모든방법에채점하며guardfallback행도분모에서빼지않는다. normal/high는보류일24h평균>=1의사후분석이다. 앞/중간/뒤는등록한floor(3*i/n)정의다. 방법선택/입력에이정답구간을전달하지않는다.

bootstrap loss는row별3seedSEdelta평균,각8blockSSE합/count,각farm4block복원재표집,pooledrowweight이며등록과맞는다. PythonMT19937·20k·seed2026100702·ties>=0·plus1p/20001·alpha.025÷6도명시됐다. 각seed전체RMSE방향과p를모두요구한다. GUARD동일이면delta0/p1/strict개선false가되므로자동fallback기각이적절하다. 개별95%CI는MSEdelta의서술CI이며6variant보정CI가아니다. 순위는평균seedRMSEdelta와bootstrap평균seedMSEdelta가다른통계임을보고한다. 8block의의존/MC오차/BLKpass1한계와원3validator미완료를유지한다.

## PS01 · P1 · scorer v2의실제실행source pin 부족

blk_score_v2는v1소스를실행시읽어문자열치환후exec한다. spec.code_sha256은__file__이v2여서wrapperSHA만기록한다. v1이바뀌어도v2SHA가같으므로실제loss/bootstrap코드가달라지는것을spec만으로발견할수없다.

필수조치: truth읽기전에v1/v2두SHA와실제transformedsourceSHA,치환대상정확1회assert를spec에기록한다. 가능하면새v3독립scorer로불필요한exec를제거한다. v1을수정해원같은파일로채점하지않는다. scorer통계명세를규칙등록과machine대조하고finite1440truth/각일24h/정답ID중복0을assert한다.

## PS02 · P1 · verified receipt 사용 시 최신 증거 pin 확인

scorer는verifiedreceipt의status/gate/predSHA/rulesSHA만대조하며receipt.codeSHA·verified_evidenceSHA의현재파일동일성을다시확인하지않는다. verification후감사파일/모듈을바꿔도scorer가받을수있다. auditstatus가PASS인것만으로그감사가현재source의실행결과임을모든경로에서확인한것은아니다.

필수조치: verify시에SG2/edge/endpoint각감사에기록한adapter/source/registrationSHA를현재파일과대조하고, scorer입구에서도receipt/verifier/evidenceSHA를pin한다. actualR3/PFN/environment/modelsettings/featuretablelineage에변화없음을확인한다. 새receipt version으로분리하며broadPFN7/8missing이나failed면targetread를차단한다. receipt가없으면추측으로채점진행하지않는다.

## 표현 범위

broadPFNpoisonbatch에앞query일부도포함된다면결과는다른질의의입력변경에대한APIbatch/queryindependence검사다. '특정cut 뒤미래입력만교란한엄밀purefuture-only검사'라고표현하지않는다. 96probe/전체모델검사는유한한실험과소스계약이며모든가능값에대한수학적증명도아니다. BLK_screen_pass는진단선별만허용하고최종adopted=False를유지한다.

PS01/PS02pin을보완하고4contextbroad감사후baselineverify→BLK진단채점으로진행할수있다. 원TM/P2LOO/EL1효과확인과최종새seed/layout판정·전체feature재탐색은계속남는다.
