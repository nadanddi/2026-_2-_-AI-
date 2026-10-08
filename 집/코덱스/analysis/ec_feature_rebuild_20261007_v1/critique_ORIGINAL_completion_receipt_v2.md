# 완료 guard v2 보완 점검

2026-10-07. v1→v2 diff·audit2 및 실제receipt/current SHA만 읽었다. 모델/정답/채점/GPU/worker변경0이다.

**OCM01의 Windows drive/colon 및 resolved root 경계, OCM02의 complete hash/parse 동일bytes 문제는 소스상 보완됐다.** manifest lexical 검사에 colon/PureWindowsPath.drive 거부를 추가하고 실제complete/output 읽기는 resolve(strict=True) 및 is_relative_to(root)를 통과한다. complete는 singleblob을 hash/jsonparse하고 끝에 다시읽은SHA와 대조한다.

저장 audit2는 unsafe11(filecallback전거부, C:/a 포함)+tamper1/actualDIAG0manifest81/incomplete66refused를 보고하며 audit/guard/raw등록3 sourceSHA가 현파일과 모두 같다. link/junction escape의 실제fault-injection PASS는 이audit에 없으므로 containment의 source 보완과 실제linktest를 구별한다.

전폴드 output이 순차검사되는 시간 동안 먼저 읽은 파일이 바뀌는 가능성까지 atomic snapshot으로 해결한 것은 아니다. 후속 strictgate는 completedimmutable producer 및 실제 소비한bytes의hash를 연결하거나 최종전수재검산해야 한다. 이는 저장완료guard 자체의 범위 제한이며 현재 fullgate아님 표시는 타당하다. 두번째 probe/source/runtimenumeric 검증을 이manifest로 대체하지 않는다.

새 핵심 storageguard blocker는 없다. exactregistry/currentregistration/source, producerstrictreceipt/freshmatrix/cache·수치·SG2·fullmixed causalgate는 별도로 남는다. full원66 rootcomplete가 없는 현재를 wholepipeline PASS로 확대하지 않는다.
