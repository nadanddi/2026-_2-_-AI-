from pathlib import Path
import re
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
report=H/'검토와_실험보고서_v1.md'
text=report.read_text(encoding='utf-8').replace('EXT10 −.431~−.488%, EXT12 −.130~−.151%. 마지막 두 범위는 정확한 CSV를 우선한다.','EXT10 −.481~−.435%, EXT12 −.151~−.108%.')
out=H/'검토와_실험보고서_v2.md';assert not out.exists();out.write_text(text,encoding='utf-8')
(H/'README.md').write_text('# 최종 사용 파일\n\n검토와_실험보고서_v2.md가 최종. v1은 문서 작성 초안이며 ST8 EXT 범위 오기는 v2에서 독립 CSV와 맞췄다. 점수/판정 변경 없음. run.py와 PROTOCOL.md는 실행 전 431826b. 큰 oof.csv는 집/코덱스/local/source_history_20261003_v1. verification_v1.json 96 및 extra_verification_v1.json 2742 checks PASS. 새 제출물/EL1/잠금 채점 없음.\n',encoding='utf-8')
cat=ROOT/'공용/데이터_단서_카탈로그.md';old=cat.read_text(encoding='utf-8');n=max(map(int,re.findall(r'\| 6\.(\d+) \|',old)))+1
entry=f'\n| 6.{n} | 미검토ST0~ST8/PH0/SE0~SE2更新·MASK안전 계열확률 잔차보정 기각 (10-03 집코덱스) | ST8공개27720행15점수독립검산기각유지;query외기역할판별/전체train분류기/outer非nested보정감사. 사전431826b SOFT_RESID10=계절v2+.1·동일온실과거nested잔차확률가중·거리감쇠,query내부현재이전만·fold内fit. DIAG+.068~+.075%/A+.086~+.098%/B−.077~−.071%/EXT10−.589~−.572%/EXT12+.069~+.073%,15칸중6개개선·p_worse .763~.796,REJECT. 公開쌍46/48·전반같은역할잔차rho F13 .331/.263 F47 .553/.504(n43/44)은탐색·物理동확정/예측효용0. 96+2742체크PASS,과거잔차접근173/360일. 검토중상대SE3별도실행확인·완료결과미검토;新PFNfit/잠금/EL1/test예측/제출0 | 집/코덱스/analysis/source_history_20261003_v1/검토와_실험보고서_v2.md, scores_v1.csv, verification_v1.json, extra_verification_v1.json |\n'
with cat.open('a',encoding='utf-8') as f:f.write(entry)
journal=ROOT/'집/코덱스/작업일지/2026-10-03.md'
with journal.open('a',encoding='utf-8') as f:
    f.write(f'\n\n## 세션 18 — 미검토 작업 업데이트와 EC 계열 잔차 비교\n- 사용자지시로집클로드10-03세션2/Git66725d1 ST0~ST8/PH0/SE0~SE2及도메인2자료검토. 연구실두곳새일지없음. 검토중46e7c92 SE3별도진행발견,아직완료결과미검토.\n- ST8공개27720행15RMSE numpy/fsum검산기각일치. 검증외기로query역할판별/전체train_X분류기·전처리/非nested보정학습문제감사. 역할은실제물리동확정이아닌대리표지.\n- predictive-modeling/analysis-verification 적용. 사전431826b SOFT_RESID10 단일비교:fold内입력역할분류기·query실내prefix확률·동일온실허용과거nested잔차만. 22fold3seed15칸 DIAG+.068~+.075%/A+.086~+.098%/B−.077~−.071%/EXT10−.589~−.572%/EXT12+.069~+.073%,기각. p .76325/.78365/.79625.\n- 독립96+2742checks PASS,scalar2736행차0·3bootstrap대조,학습buffer/과거·미래prefix불변감사. 과거가용173/360일;공개전반쌍잔차상관존재는사후탐색·일반화미확인. 카탈로그6.{n}.\n- 現실제W40G·EC계절v2유지.新PFNfit/원시EC라벨/잠금/EL1/test예측/제출없음. own백그라운드종료,ClaudeSE3진행중이므로전체프로세스종료로표시하지않음. 次=SE3결과확인·fold가용짧은과거잔차와후반전이/분류오류分解.\n- 파일:analysis/source_history_20261003_v1/PROTOCOL.md/run.py/검토와_실험보고서_v2.md/README/scores_v1/result_v1/ST8_independent_scores_v1/segments_v1/verification_v1/extra_verification_v1,local동명/oof.csv. v1문서초안ST8 EXT수치범위는v2에CSV일치정정·판정변경0.\n')
handoff=ROOT/'공용/HANDOFF.md';s=handoff.read_text(encoding='utf-8');needle='# HANDOFF — 지금 상태 (세션마다 갱신)'
new='\n\n> **最新 2026-10-03 집·코덱스:** 미검토 구조/주기/계절/잔차 작업6.207~6.211 검토, ST8공개15점수재계산기각유지·query외기/분류기fit/非nested감사. 사전431826b MASK안전 SOFT_RESID10 계열확률×과거nested잔차 단일비교 기각(DIAG+.068~+.075%,p .763~.796). 96+2742검산PASS. 실제W40G·EC계절v2유지,새제출/잠금/EL1채점없음. 코덱스프로세스종료,상대Claude SE3(46e7c92)는진행중확인·완료결과미검토. 상세 집/코덱스/analysis/source_history_20261003_v1/검토와_실험보고서_v2.md. 次:SE3완료결과의가용입력감사·fold가용과거잔차/후반전이분해.'
handoff.write_text(s.replace(needle,needle+new,1),encoding='utf-8')
checks=ROOT/'공용/확인기록.md';s=checks.read_text(encoding='utf-8');lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('| 집 · 코덱스 |'):
        lines[i]='| 집 · 코덱스 | 2026-10-03 | 기존2fbfcd4検討+同日집클로드세션2・Git66725d1 ST0~ST8/PH0/SE0~SE2/6.207~6.211更新,자료2건읽음. ST8공개27720행15점수numpy/fsum再計算기각일치,query외기·전체train분류기/전처리·非nested감사. 안전SOFT_RESID10기각·96+2742검산PASS. 원시EC/잠금/EL1再채점0. 연구실두폴더최신10-01·추가일지없음. 검토중46e7c92 SE3진행확인,완료결과미검토. 최종보고서 집/코덱스/analysis/source_history_20261003_v1/검토와_실험보고서_v2.md |'
checks.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('saved catalog',n)
