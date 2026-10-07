# 원66fold feature 준비 v2 등록 독립 재검토

2026-10-07. registrar/runner v2와실제등록을읽고PowerShell로109sourceSHA를직접대조했다. 모델fit/GPU/정답값열람/채점/준비runner실행0. 승인서비스실패로중단된ML실행을다른경로로재시도하지않았다.

## 판정

OP01 핵심보완은소스·등록상닫혔다. 새feature-preparation core blocker는발견하지못했다. 이는등록된feature준비에대한판단이며실제66fold준비완료또는모델검증PASS가아니다.

## 실제pin·범위확인

109source 현재불일치0,runner와registry currentSHA가등록과일치했다. exact24familymap,DIAG10 10/P2LOO46/EL110fold,query행수8640/1104/1104,TM111일2664행,fold당6prefix/총396이기록됐다. model_fit_registered=false이다.

registrar는원foldregistry와query집합·fold키를비교하고integrityPASS/registrySHA,sortedunique IDs/train-query-forbidden disjoint,전체9600IDpartition,train-query recorddisjoint와각record24h완전성을검사한다. source를수집한후다시전부SHA확인한다. 고정originalregistry에서mixedinputframe을daylocal로처리하는전제가보다명시적으로봉인됐다.

runner resume는savedexactschema/validator-fold/1187/24matrix열/hash형식/emptylist부분집합을검사한다. fresh/resume prefix합계를실제로누적하고expected66filename집합및실제디렉터리foldfiles집합과일치시켜396을확인한다. hardcoded396범위표현을실제검사합계에연결한점은타당하다.

## 유지해야할한계

resume의payload/hash형식검사는실matrix재계산이아니다. 아직signature에별도shape필드는없지만rowIDs/columns와등록float64dtype가binarySHA의의미를규정한다. 미래모델runner는새로matrix를만들어train/querySHA/행·열순서·shape/dtype를대조하고실제imputer/FPNcache/전체모델boundary를별도검증해야한다.

이준비는reference공개EC/sub_temp만로드하며query정답은파싱하지않는다. 3rid/fold prefix표본396비교가66fold의모든입력시점에대한독립증명은아니다. 기존MASK/test_XNaN및purge/locked정책의원registry/integrity근거를유지한다.

CPUcached baseline의원fold모델정책등록은별도이며기존submission14/역사적GPU와동일하다고할수없다. all24×3seed원TM/P2LOO/EL1효과·최종미사용seed/layout1회및전체목표는아직남는다. 새등록준비가BLK48순위로원24후보를제외하는근거를추가하지않는다.
