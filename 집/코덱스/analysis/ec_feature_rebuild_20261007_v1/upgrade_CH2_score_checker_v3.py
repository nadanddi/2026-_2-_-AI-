from pathlib import Path
import ast
here=Path(__file__).resolve().parent
source=(here/'crosscheck_ch2_BLK_score_v2.py').read_text(encoding='utf-8')
marker="    reports=[];rmse_checks=0"
source=source.replace(marker,"""    expected_variants={(scope,method) for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS']
                       for method in ['CH2_REFONLY_GUARD','FLANK_SOURCE_MATCH','PAST_QUERY_PREFIX_STATE']}
    assert len(result['results'])==6 and {(v['scope'],v['method']) for v in result['results']}==expected_variants
    expected_segments={'전체','일반','고EC','앞','가운데','뒤','F13','F47'}|{f'hour{h:02d}' for h in range(24)}
    expected_cells={(seed,segment) for seed in [47,1414,6464] for segment in expected_segments}
"""+marker)
source=source.replace("            assert len(item['cells'])==96", "            assert len(item['cells'])==96 and {(v['seed'],v['segment']) for v in item['cells']}==expected_cells\n            overall_deltas=[]")
source=source.replace("close(cell['delta_RMSE'],cr-br);rmse_checks+=2", "close(cell['delta_RMSE'],cr-br);rmse_checks+=2\n                if cell['segment']=='전체':overall_deltas.append(cr-br)")
marker="            # Algebraically factor the SSE difference independently"
source=source.replace(marker,"            assert len(overall_deltas)==3\n            close(item['mean_seed_delta_RMSE'],sum(overall_deltas,Decimal(0))/3)\n"+marker)
source=source.replace("'RMSE_checks':rmse_checks,", "'RMSE_checks':rmse_checks,'exact_variant_cell_sets_checked':True,'summary_mean_checks':6,")
ast.parse(source);out=here/'crosscheck_ch2_BLK_score_v3.py';assert not out.exists();out.write_text(source,encoding='utf-8')
print('Checker3 exact6variants/576cells +6independent mean checks added; statistics unchanged')
