# 원66 domain 조립1 사전 독립 비평

2026-10-07. assembler1 소스만 읽었다. 등록/조립/모델/정답/채점/GPU/worker변경0이다.

고정recipe와채점차단순서는타당하다. 그러나 **사용자의 폴드별 재개 요구를 현재초안은 충족하지 않으며 singlewriter도 없다. 실제조립을 시작하기 전에 새버전에서 보완해야 한다.** 원producer는수정하지 않는다.

## OA01 — 폴드별 strict resume·singlewriter

현재existingfolder assert는이전결과보존에는안전하지만 첫실행이중간폴드에서중단되면완료한폴드도건너뛸수없다. 66fold와fresh81/4context를계속처리하는본작업의재개요구와어긋난다. 두writer는동시에 expensivefresh계산을하고늦게 folder.mkdir에서충돌할수있다. rootcomplete쓰기도explicit독점조건이없다.

등록SHA/PID/start_ticks를담은 exclusive lock을expensive처리전에생성하고finally에서자신token만제거한다. stale lock은실제process·starttime/다른writer부재확인뒤새artifact로보존한다. strictresume는완료된prediction+audit pair/exactschema/ID·75stage·3seed×24candidate/fullstagefinite·producer/manifests/choice/source/metadataSHA를확인하고freshinputs/expectedcontract와묶은뒤만수용한다. 미완성pair는보존·거부하고자동덮어쓰기하지않는다. 최종132 exactmanifest도등록된규칙대로재개비교한다.

## OA02 — 최종소비·출력lineage 재확인

원raw/PFNoutput총files는최종전수SHA재검산하지만raw/PFNrootcomplete/foldcomplete각SHA는초기manifest로기록한뒤끝다시검사하지않는다. assembled132도outputs에첫저장SHA만넣고finalcomplete전current전수대조하지않는다. 끝에서source/root·foldcompletion·소비output·자신132파일모두현재일치와exact예상집합을확인한다. atomic여러파일쓰기만으로폴더snapshot이되지는않는다.

futureverifier/fullgate는 pipeline등록뿐아니라rawdriver4/PFNdriver3등록과actualassembler/source/manifest/context/weights까지transitivepin해야한다. 현재등록기가없으므로그증거가이미완료된것으로말하지않는다.

## 수용한모델·누수경계

원66raw5346/PFN528exactcompletionmanifest를feature/yload전에확인한다. productionloader2all66prepared·fresh24matrix, validator2raw81/PFN4freshmatrix/finitey/median/contextID/strictcachetrace를묶고actualvalidationblobSHA가manifest와같은지대조한다. lower/upper는foldtrainy min/max다. 평가정답API/parse없고ctxreference는정확foldtrain이며query/gap과분리된다.

25 ET안×3seed=75stage, LGB/MLP/PFN그foldbaseline고정, kernel2rawmix→shrink한번→trainclip→RAW_PASS SG2→clip은등록recipe다. choice는prediction값과독립이라는기존source분리근거위에서한plan으로reuse한다. baseline각row×3seed의sourceaudit와raw81/PFN4count가있다. candidate별actualprefix sourcecomparison/독립산술/freshfuturepoison은여기없지만saved/audit/complete의wholefalse/NOSCOREGATE에정확히명시한다.

actualmodulepath loop는 **7개**(features,validation,completion,kernel,sgplan,rawproducer,pfnrules)다. 설명의8개와현재소스는다르며baseline/checkpoint/sg1·sg2모듈도새등록 runtimebinding에명시하면좋다. rawproducer.runtime_paths 및SGplan핵심소스체크가간접보완하지만'8개직접검사'라고보고하면안된다.

storagepair경로는as_posix라앞선Windows집계오류를피한다. 다만savedstagearray의finite·exactkeys·bounds·candidateidentity를futuregate가엄격검사해야한다. kernelfullvectors는streamingprefix-only접근API아니며실SG2choicecache를fresh로만든inputpoison을전수규칙에맞춰검사해야한다. baselineSG2 sourcechecks횟수만candidatecausal증거로확대하지않는다.

현재source초안은성능·원전체검증기모델완료를증명하지않는다. 원24전수·DIAG/TM고정통계교집합/모든seed×원검증기방향·최초미사용1회/의미있는선별최종정리파일검증뒤goal종료는남는다. 중간조립결과로순위선별하거나정답parse를열수없다.
