# IRR4 첫 fold 독립 중간 검토

2026-10-09. critic 별도무학습실제검산exit0. rawCSV수치정답접근0/EC정답0/외부자료0/fit0.

판정: **동일등록run_irr_v2로나머지11fold를계속할수있다.** IRR등록source/runtime SHA가현재파일과같고parent_recipe_SHA가원CODEX등록SHA에연결된다. 첫fold의현재NPZ SHA/parentNPZ SHA/각metadatarecipeSHA는실제bytes와같다.

IRR NPZ의query960 IDs가원parent NPZ와두metadata에전수같다. IRR저장BASELINE배열은원parentBASELINE과raw np.array_equal이고baseline재fit프로브receipt는0이다. 직접검산은저장배열복사동일성과receipt0연결이며critic이baseline을별도학습한것은아니다. candidate960예측은shape가맞고전부유한값이다.

metadataweights분포/전체seasonnotes는원parent와같다. train8304×93=772272 mask동등셀receipt가맞다. 원89+phys19 및6,388,464셀globalfeature감사는root현재실행assert통과로그증거이며critic이특징을별도재생성하지않았다. 이셀합을독립관측수/전체모델규정검사로표현하지않는다.

전체resume이초기같은feature감사를다시실행해도모델의독립확인횟수가늘어난것은아니다. firstfold는cache 재사용하고나머지11candidatefit만새로실행한다. 최종13pipelinefit은baselineprobe1+candidate12이며누적/현재processfit를구별한다. 이번승인은구성원진단범위이고채택/전체W40GS/제출진행이아니다. full결과후원parentBASELINE전수/ID·purge·해시/RMSE/p/CI와사전screen판정을검산한다.
