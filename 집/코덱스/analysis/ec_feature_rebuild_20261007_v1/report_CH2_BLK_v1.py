from pathlib import Path
import json,csv,hashlib,datetime,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    result=read('CH2_BLK_diagnostic_results_v1.json');check=read('CH2_BLK_score_independent_crosscheck_v1.json')
    assert check['result_sha256']==sha(HERE/'CH2_BLK_diagnostic_results_v1.json')
    assert check['status']=='DECIMAL60_ALL_CH2_RMSE_FACTORIZED_LOSS_BOOTSTRAP_PASS' and check['RMSE_checks']==1152
    assert len(result['results'])==6 and not any(v['BLK_screen_pass'] for v in result['results'])
    assert (HERE/'critique_CH2_scored_diagnostics_v1.md').exists()
    lines=['# CH2 고정 이웃 사슬 BLK 진단','',
           '사전 고정한 6개 비교는 모두 통계 기준을 통과하지 못했다. 두 방법은 세 시드 모두 오차를 줄였지만, 효과가 8개 중 같은 한 블록에만 나타났다. 이 결과로 CH2 전체 또는 이웃 원자료의 무용함을 주장하지 않는다.','',
           '## 고정 비교와 결과','',
           'CPU reference-only cached 기준선, BLK 1,440행·60일·8블록, 시드47/1414/6464. 주 비교는 QUERY_ROLE, RAW_PASS는 적용 범위 진단이다. 기존6+도메인48+이번6=누적60비교, P(worse)<.025/60을 실행 전에 고정했다. 이전 도메인 등록의 alpha54는 수정하지 않았다.','',
           '| 범위 | 방법 | 평균 시드 RMSE 변화 | P(worse) | 판정 |','|---|---|---:|---:|---|']
    for v in result['results']:
        lines.append(f"| {v['scope']} | {v['method']} | {v['mean_seed_delta_RMSE']:+.10f} | {v['p_worse']:.8f} | 진단 FAIL |")
    lines+=['',
            'CH2_REFONLY_GUARD는 지원0행이라 기준선과 같았다. FLANK_SOURCE_MATCH는25행, PAST_QUERY_PREFIX_STATE는14행을 지원했다. 활성4비교의 block SSE 변화는 index3에서만 음수이고 나머지7개는0이다. 재표집에서 그 블록이 빠지면 변화0이므로, 0을 악화와 함께 세는 고정 통계에서 p≈.319이다.','',
            '## 검증과 한계','',
            '- 94개 등록 pin·현재98개 gate pin을 확인했다. 실제360개 forbidden-query 교란/역순 검사, 모든1,440행 prefix 재생,1,560회 소비 로그,25,920개 독립 끝점 산술 검사를 통과했다. 산술 최대차1.11e-16.','- Decimal60 독립 검산은1,152 RMSE·48블록SSE·6 bootstrap p와 CI·576셀 고유키·6요약평균을 확인했다.','- 일반1,368행/고EC72행3일과 앞·가운데·뒤, 농장,24시간별 효과를 사전에 고정한576셀 CSV에 보존했다. 이 사후 구분으로 적용 규칙을 바꾸지 않는다.','- BLK는모두pass1이고 고EC3일뿐이다. 기준선은과거GPU출력과의바이트동일성을주장하지않는CPUcachedrecipe이다. 정답을 이용한 사슬은 물리적인 온실 동이나 실제 시간 순서를 증명하지 않는다.','- 원TM111/P2LOO/EL1 검증과 최초미사용seed/layout1회, 도메인·문헌·전체데이터 단계, 최종 보고서는 남아 있다. 이번6안은튜닝없이종료하며전체목표는계속진행한다.','',
            '## 보존한 실패와 보완','',
            '등록v1의 CSV 열 이름 오류와 partialCSV6을 보존했다. v2 등록 뒤 query adapter의 전체 schema 선검사·실제 import경로를 보완한v3를 등록한 후추론했다. verifier2의tuple/list직렬화비교실패는실제row600에서원인확인후새verifier3로JSON정규화비교만수정했다. 예측·통계기준은변경하지않았다. score spec2는추론후·정답채점전에pointer수정되었으며created_before_inference=False를기록했다(등록script의기존콘솔문구는이시점과맞지않으므로사용하지않는다).','',
            '## 근거','',
            '- CH2_BLK_registration_v3.json / feature_candidates_v8.csv','- checkpoints/CH2_BLK_v1/predictions.json / audit.json','- CH2_BLK_verified_gate_v3.json / CH2_gate_serialization_diagnosis_v1.json','- CH2_BLK_diagnostic_results_v1.json / CH2_BLK_diagnostic_cells_v1.csv','- CH2_BLK_score_independent_crosscheck_v1.json','- critique_CH2_actual_gate_v1.md / critique_CH2_scored_diagnostics_v1.md','']
    out=HERE/'CH2_BLK_진단보고서_v1.md';assert not out.exists();out.write_text('\n'.join(lines),encoding='utf-8')
    with (HERE/'feature_candidates_v8.csv').open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    matched=0
    for row in rows:
        for v in result['results']:
            if row['candidate_id']!=f"CH2_{v['scope']}_{v['method']}":continue
            row['status']='SCREEN_FAIL_NO_ADOPTION' if v['method']!='CH2_REFONLY_GUARD' else 'INACTIVE_BASELINE_IDENTICAL_NO_CHAIN_REJECTION'
            row['performance']=json.dumps({'mean_seed_delta_RMSE':v['mean_seed_delta_RMSE'],'p_worse':v['p_worse'],
                                          'alpha':.025/60,'reference':'CH2_BLK_diagnostic_results_v1.json',
                                          'original_validators_tested':False},ensure_ascii=False);matched+=1
    assert matched==6 and len(rows)==202
    out=HERE/'feature_candidates_v9.csv';assert not out.exists()
    with out.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    catalog=ROOT/'공용/데이터_단서_카탈로그.md'
    last=max(int(n) for n in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))
    number=last+1
    note=f'''\n### 2026-10-07 집 코덱스 CH2 실제 진단 완료·원66/도메인조립 재개 (checkpoint v17)\n- CH2 exact3×2scope×3seed 등록3/94pins/CSV202,8lazy-context+48prefixsynthetic PASS. actualrunner31873 terminalexit0;360query경계검사. verifier2 tuple/list mismatch 실패보존→원인JSONexact확인→verifier3 actualexit0/25920scalar max1.11e-16/1440prefix/1560consumption/98pins. 독립gate비평PASS.\n- score2+Decimalchecker3 실제exit0:6안전부FAIL alpha.025/60. QUERY_ROLE FLANK -0.00119145098/p.318684,PASTQUERY -0.000280008123/p동일; RAW -0.00106063236/-0.000223016144. guard0행/p1;지원25/14행·같은1block에효과집중. 1152RMSE/48SSE/allpCI+독립사후비평PASS·튜닝/채택0. 보고서/CSVv9 저장·카탈로그6.{number}.\n- 원66특징prep default는pandas파일sandbox접근거부로version확인전exit1(모델0/저장폴드0). 정식require_escalated승인성공→session77114 actualrunning/DIAG10fold0 PASS 확인. 승인경로현재성공을근거로원등록도메인assemble 동일정식경로재요청승인성공→session55500 actualrunning. 이전credits미실행상태는이전시점이며새실행을우회하지않음.\n- 다음원77114/55500을직접poll;완료/누락handle확인없이재시작금지. 도메인조립완료후등록verify→actualgate독립비평→score/Decimal;원66prep완료후modelrunner전체24×3seed준비. 문헌24/데이터142/최초미사용seed-layout1회/전체최종보고서남음·goal active/GPU·제출0.\n'''
    record={'checkpoint_version':17,'recorded_at':datetime.datetime.now().astimezone().isoformat(),
            'previous_goal_turn':'progress: prefix48/training reference/fresh file replay',
            'CH2_status':'six actual diagnostics screenFAIL, independently verified','CH2_scorer_terminal_exit':0,
            'CH2_checker_terminal_exit':0,'CH2_RMSE_checks':1152,'CH2_adoption':False,
            'domain_original_preparation_session':77114,'domain_original_preparation_last_authoritative_status':'running; DIAG10fold0 PASS',
            'domain_assembly_session':55500,'domain_assembly_last_authoritative_status':'running; formal escalated approval accepted',
            'approval_failure_currently_reproduced':False,'approval_error_resolution_evidence':'distinct originalprep approval succeeded; original domainassemble formal retry accepted',
            'whole_goal_complete':False,'GPU_used':False,'submission_created':False,'catalog_number':f'6.{number}'}
    out=HERE/'checkpoint_record_v17.json';assert not out.exists();out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    progress=HERE/'PROGRESS.md';old=progress.read_text(encoding='utf-8')
    progress.write_text('# 현재 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v17)\n'+note+'\n아래는시점별과거상태.\n\n'+old,encoding='utf-8')
    for path in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
        with path.open('a',encoding='utf-8') as handle:handle.write(note)
    with catalog.open('a',encoding='utf-8') as handle:
        handle.write(f'\n| 6.{number} | **CH2 실제6비교 전부진단FAIL·효과1block집중; 승인경로 성공 후 원66/도메인조립 실제재개** (2026-10-07 집 코덱스) | exact3method×2scope×3seed/누적60보정. 등록94pin/currentgate98일치·실360경계/1440prefix/1560소비/25920산술max1.11e-16 PASS. score+Decimal actualexit0/1152RMSE·48SSE·6pCI·exact576cell/6mean PASS. QUERY_ROLE FLANK -0.001191451,p.318684;PASTQUERY -.000280008,p동일;RAW -.001060632/-.000223016. 활성25/14행·8중1blockSSE만감소;guard0행/p1. sixFAIL alpha.025/60·사후비평/튜닝0·CH2전체/물리source기각금지. 등록CSVschema오류/tupleJSON검증실패보존·새버전보완,통계/예측수정0. 원66prep default pandas접근거부exit1뒤정식승인성공 session77114 running/첫foldPASS;이근거후원domainassemble formalretry 승인성공 session55500 running,이전credits실패는과거상태/우회0. 전체goal미완료/원검증기/문헌/데이터/최종1회남음/GPU·채택·제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/CH2_BLK_진단보고서_v1.md;CH2_BLK_diagnostic_results_v1.json;CH2_BLK_score_independent_crosscheck_v1.json;critique_CH2_scored_diagnostics_v1.md;checkpoint_record_v17.json |\n')
    print('CH2 verified report, CSV9, checkpoint17 and shared logs saved; goal active')

if __name__=='__main__':main()
