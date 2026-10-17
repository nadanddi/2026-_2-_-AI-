# IRR4 코드 실행 전 독립 비평

2026-10-09. run_irr_v2.py 정적검토, critic fit0/외부자료0.

판정: **--first 실행 차단점 없음**. 원run_v2/runtime SHA=등록SHA,등록DATA=원preparation Pin=현재actualCSV SHA로출처가연결된다. 등록원89/물리19도원정의와같다. firstbaseline fit은원89열/물리19/기존LGB설정/seed9109·foldweights/season에서학습하고원cache와raw np.array_equal을요구하여허용오차추가없다.

특징은원CODEX frame에raw rad_eff와farm/day내hl1/3/8 EWMA4열만추가한다. 원89+phys19 exact동등/각day24행·24uniquehour검사및원build_features hour0..23검사가시간격자를고정한다. future globalT/otherfarm/shuffle/officialtestMASK와foldtrain외입력NaN93열동등감사가실행전·fold중각각있다. 라벨은RAW의sub_temp만읽고통계는코드사전ALPHA=.025/3와CI같은quantiles를사용한다.

parent12fold recipeSHA/NPZsha/queryIDs가대조되고weightscounts·전체seasonnotes가parent와exact같다. 불변weights 순서개별값해시까지는없지만동일원weights함수/동일trainIDs/동일rawSHA에서생성되고첫baselineexact로연결된다. 캐시의원baseline raw가새NPZ에복사되어최종독립검산에서parentBASELINE배열과전수비교할수있다.

주의: adjust=True/ignore_na=False는CODEX원reset의pandas기본규약에맞춘정의이며BASEseg ignore_na=True와같은EWMA라고하지않는다. 'NaN유지'는raw rad_eff결측이보존된다는뜻으로정확히쓴다. pandas EWMA는현재raw가NaN이어도과거값이있으면평활값을반환할수있으며ignore_na=False는결측칸시간의가중치처리다. 이것은미래보간이아니다.

첫fold전달후실제감사셀/NPZ/등록SHA·parentbaseline전수exact·maskcells/train/query/purge/weights/season/유한candidate를별도검토한다. 코드에assert가있다는정적증거와실제assert통과를구별하며아직효과판정은하지않는다.
