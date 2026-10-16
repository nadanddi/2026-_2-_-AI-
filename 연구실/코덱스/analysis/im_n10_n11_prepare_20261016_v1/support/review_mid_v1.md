# 구현 중간 비평

build_v2.py/validate_v3.py만검토했다. N10최종return직전 ff5유한/gap전체정밀도/dg1 strict> 조건으로max.6, N11 p<.8만삭제는계획과일치한다. 모델/기타파일SHA보존과원본사후SHA검사, 독립두패키지구성도적절하다.

필수검사보완1건: validate_v3.py의pred[9/10/11] 및expected/base에 유한성/[0,1] assert를추가해야한다. 현재는NaN>=.5가False이므로일부진짜행의NaN을판정일치로놓칠수있고,max_abs=NaN도json에기록될수있다. 모델규칙변경없는검사보완이며실행전새버전또는실행후저장배열별도필수검사로충족가능하다.

동일백본1회+각패키지실제_final호출은모델SHA가같고차이가_final에한정돼있으므로적절한연산재사용이다. 단각패키지predict전체를독립실행한검사로표현하지않는다. 예정된각ZIP새압축해제공식self_check가이를보완한다.

추가미세검사: 압축전총파일크기<1e9와ZIP크기는다르므로실제zip_bytes<1e9도확인한다. cache경보조건의strict>와nextafter>=동일성assert는좋다. E4보류/원본DB부재오탐가능설명도유지한다.
