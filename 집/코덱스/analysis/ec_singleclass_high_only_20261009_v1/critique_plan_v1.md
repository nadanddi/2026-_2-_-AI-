# 단일class 고EC학습 분류 독립 계획 비평 v1 — 2026-10-09

검토: PLAN_v1.md/run_v1.py. 사용자확정은EC수치회귀가아닌고EC여부분류이며앞선회귀캐시감사의추가작업은중단한다.
판정: 기존73현재prefix특징/ET설정을두고양성만학습하면어떻게되는지짧게확인하는실험은타당하다. 모든26기존LOO분할의class검산과실제대표1fold/3seed만fit한다는범위를분명히한점이좋고prepare+firstseed를진행할수있다.

source/학습: 공개360일metadata에서LOOtrainindices중high1만남긴다. 실제첫fold는high20일480행,heldhigh1일24행과ordinary334일8016행은train과교집합없음을assert한다. medianimputer도새고ECtrain에만fit하며day/정답은73특징에없다. 이전73특징은day별prefix라일반날의다른날입력이highday특징으로섞이는구조가아니다. sourceSHA와기존P.check가원자료/코드변경을확인한다. root/critic는별도루프에서26고ECsubset과행/일분모를검산할것.

단일class의의미: 이ExtraTreesClassifier는학습classes_=[1]이고positive_score가그유일한확률열을읽으므로모든입력score1은예상되는정의적퇴화다. heldhigh탐지100%를성능개선이라고부르면안되며미사용ordinary오탐100%와함께제시해야한다. score1은현실고EC확률100%로보정된추정이아니다. 이ETsupervisedbinary의결과를모든positive-only밀도/이상탐지기법까지일반화하지말것.

분모/검증: heldhigh은1농장·1일/24시간이며3seed가새양성3일을만들지않는다. ordinary334일×24prefix는모두학습에서빠졌지만독립8016농업사건이아니다. 전score/prediction1이면day단위로도고EC1/1·ordinary334/334를직접확인할수있다. 실제26LOO전체학습은하지않으므로'26fold LOOCV 성능100%'라고표현금지다.

마무리범위: firstseed후독립중간검산·나머지2seed·최종비평만필요하다. 새104fit/새최종모델저장/threshold튜닝/EC보정·제출은없다. 이전양class분류기나양성LOO결과를변경하지않고퇴화확인으로한정한다. 추가필수코드수정은없다.
