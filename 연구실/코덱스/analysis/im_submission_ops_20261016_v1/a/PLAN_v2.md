# A2 토큰 전문가 실행 계획 v1

사용자지정D1분기37/181<.30으로A2만진행.나단9+새토큰전문가한개, C1과혼합안함.14시슬롯/패키지+self_check13:30마감.실패후파라미터수정·다른후보전환없음.

## 고정 새 정보
CLIP ViT-B/16의기존224px조각54개와그대로같은백본/정규화사용.각조각14×14토큰(16px셀),최종transformer뒤ln_post의patch token768차원(proj전;CLS제외).백본가중치학습0.기존추론의B/16 pass를hook으로재사용하므로새백본pass추가0,선형점수계산만추가.사진점수는54×196토큰전체로지스틱점수최댓값.

## 학습 고정
자료=h3진짜1800+h3j진짜1800+h3부분스타일501/v7 393+h3s작은부분생성450.원본mask저장경로검사.진짜token=0,부분가짜token16×16셀의저장마스크평균≥.3이면1,정확0이면0,0~.3중간은제외.
사진별고정rng(crc32('A2S|dataset|file'))로진짜최대32token,가짜양성최대64/배경음성최대32token추출(전체54조각토큰에서표집).균등표집/첫픽셀순아님.표집점수나검증결과로샘플추가안함.
StandardScaler+LogisticRegression(C1,balanced,max_iter3000).group5fold(같은source/group모든파생동일foldassert)로학습진짜사진점수OOF생성.표준화도fold학습token만fit.
θ=표준JPEG게이트안진짜사진의99.75%분위,풀=학습진짜OOF+newreal/realg/realn배포점수.h3j는음성계수학습에포함하되게이트밖은θ풀제외(운영모집단한정,실행전선택).rc별추가θ없음(사용자A2지정단일.25%).
계수학습에R2~5/공개/jig500/sealed/realg/realn없음.newreal/realg/realn문턱보정사용하므로완전독립시험이라하지않음.

## 결합·판정
표준JPEG게이트안p_A=max(p9,g(token_max,θ));밖은p9유지.다른전문가/θ/기본판별기불변.
필수:①R2~5합fixed_fake>new_FP②sealed3종new_FP≤1,jig500 FP증가≤1%p,pub21/21③R3/R4/R5각BA≥N9.θ≥1/≤0/비유한·학습미수렴등수치실패는불통과(사후C/θ수정없음).평가참고로처리진짜5종/GPT류성능보고,공통필수에새조건추가안함.
PASS만나단11(나단9+A2)CSV/ZIP/한국어설명+self_check+Colab요청.확증채택아닌가설검증제출후보.T4실측미확인상태를제출준비완료라단정안함.

## 파일/검증
코드·분석은연구실코덱스analysis/im_submission_ops_20261016_v1/a, 서버동명a/, 무거운자료ownlocal동명.샘플token만저장하며전token100GB캐시는안만듦.고정B16샘플추출1pass→LRfit→필요사진토큰max계산1pass(OOF/final모델동시에선형계산)으로순서실행.C1 GPU작업끝난뒤GPU0무료확인후시작.
계획비평/첫샘플·fit중간비평/최종비평.CLS출력hook전후불변, 16px mask격자/토큰순서검사, 사진점수재생, array독립집계, 기존N9baseline대조, 원모델SHA불변, 깨끗한추출self_check.

