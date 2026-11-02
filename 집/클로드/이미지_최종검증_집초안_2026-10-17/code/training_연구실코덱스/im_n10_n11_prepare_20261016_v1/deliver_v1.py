"""Finalize approved local deliverables only after all required checks exist."""
import json,csv,hashlib,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];LOCAL=ROOT/'연구실/코덱스/local/im_n10_n11_prepare_20261016_v1'
read=lambda n:json.loads((P/n).read_text(encoding='utf-8'))
upper=read('build_result_upper_v1.json')['10'];eleven=read('build_result_v2.json')['11'];sc10=read('self_checks_upper_v1.json')['10'];sc11=read('self_checks_v1.json')['11'];v10=read('validation_r2r5_upper_v1.json');v11=read('validation_r2r5_v1.json')
for n,build,sc in ((10,upper,sc10),(11,eleven,sc11)):
    assert not sc['errors'] and sc['info']['sample_metrics']['accuracy']==1 and sc['info']['run']['order_check']['consistent']
    arc=LOCAL/f'팜모니_나단{n}.zip';assert hashlib.sha256(arc.read_bytes()).hexdigest()==build['zip_sha256'] and arc.stat().st_size<1e9
    vals=v10 if n==10 else {s:d['11'] for s,d in v11.items()}
    assert all(r['decision_mismatches']==0 and abs(r['package_ba']-build['metrics'][s]['candidate_ba'])<1e-15 for s,r in vals.items())
    folder=ROOT/'제출/미제출'/('이미지_20261016_나단10_dg1' if n==10 else '이미지_20261016_나단11_ppre제거')
    instruction='나단9의 마지막 출력에서 ff 5값이 유효하고 gap≥3.4739584922790527, 1.0911458730697632<dg1<2.3890950679779053이면 p=max(p,0.6). JPEG 형식과 무관하다.' if n==10 else '나단9의 진짜 확정 분기에서 ppre<0.8 조건 하나만 제거했다. 가짜 우선 분기, rc 문턱과 후단 전문가, JPEG 게이트를 그대로 유지한다.'
    risks='R2~R5 가짜76행을 고치고 진짜4행을 새로 틀린다. 처리 진짜 신규오판은 jig1·area/png/c444 각3·webp1이며 보호 진짜3종은3행이다. 원본 제외 시뮬레이션에서 원본만 제외한 진짜 경보는 상한 전823/2685→상한 후3/2685다. 손해0 보장은 없다. 기존 “처리 진짜 신규오판+0” 조건을 충족했다고 주장하지 않는다.' if n==10 else 'R2~R5 진짜10행을 고치지만 가짜2행(filter_sp·filter_jpeg)을 새로 놓친다. 처리·흔들기·보호 진짜 신규오판은0이다. 전문가가 진짜 확정 뒤 다시 가짜로 바꿀 수 있으므로 진짜 확정 수와 최종 개선 수는 다르다.'
    lines=[f'# 팜모니_나단{n} — '+('dg1 상한판' if n==10 else 'ppre 제한 제거'),'', '**가설 검증 제출 후보이며 아직 업로드하지 않았다. Colab T4 실측 대기다.**','',instruction,'','모델 가중치·AI Hub DB·나머지 문턱과 규칙은 나단9 그대로다. 나단10과11을 섞지 않는다. E4 시험521장 입력 통계는 사용자 보류로 미사용.','','가설 검증 제출, 효과는 시험 사진 원본이 AI Hub DB에 있을 때만, 위험 = 운영진 진짜 재처리 시 오판.','','위 문장은 기대효과의 가정이다. 실제 원본 존재를 인증한 것이 아니며, 원본이 DB에 없어도 최근접 지문 규칙이 발동해 오탐할 수 있다.','','## 검증과 위험','',risks,'','| 자료 | 나단9 균형 정확도 | 이 후보 균형 정확도 |','|---|---:|---:|']
    for sn in ('im_set_r2','im_set_r3','im_set_r4','im_set_r5'):
        r=build['metrics'][sn];lines.append(f"| {sn.replace('im_set_','').upper()} | {r['base_ba']:.8f} | {r['candidate_ba']:.8f} |")
    lines+=['','R2~R5는 이미 개봉한 탐색 자료다. 실제 패키지와 기존 진단의 판정·균형 정확도가 모두 일치했다. 새 압축 해제 사본의 공식self_check PASS, 공개21/21, 순서 독립성 통과. 서버RTX6000 Ada 공개21장의 시간은 T4 전체 실측이 아니다.','',f"ZIP 압축 해제 크기 {build['size']:,}B(<1e9), ZIP {build['zip_bytes']:,}B. SHA256 `{build['zip_sha256']}`.",'','## 파일과 T4 확인','',f'실제 제출·재현 ZIP: `C:/work/farmai/연구실/코덱스/local/im_n10_n11_prepare_20261016_v1/팜모니_나단{n}.zip`. ZIP 안에 inference.py·model/·README·requirements가 모두 있다. 원본 N9 ZIP과 상한 없는 구 N10 ZIP은 제출하지 않는다.', '',f'같은 폴더의 `이미지데이터제출용_나단{n}.csv`는 운영진 제공 CSV의 바이트 동일 사본이다. 숨김 사진 예측 CSV가 아니다.','', '공통 `colab_time_v1.py`를 두 ZIP과 함께 T4 환경에서 실행해 전체521/524장 로드+예측 시간을 확인한다. 정답은 읽지 않으며 E4 중간값 통계도 계산하지 않는다. 각 후보300초 이내인지, 실제GPU=T4인지, 전체사진수인지 확인한다. 정확도와 LB는 이 시간 검사로 알 수 없다.','','## 최종 운영','', '**18시에 LB≥0.98 후보 중 최고를 사용하고, 없으면 나단9를 재제출한다.** 같은 최고점이면 사전 고정한 나단10 우선. LB로 문턱·규칙을 수정하지 않는다. 최종은 마지막 유효 제출이며 업로드는 사용자. 현재 계정에는15시알림이 없다. 남은 제출은 마지막 사용자 확인3회이며 이번 작업의 업로드0회다.']
    dest=folder/f'설명자료_나단{n}_20261016.md'
    with dest.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    shutil.copyfile(P/'colab_time_v1.py',folder/'colab_time_v1.py')
    (folder/'실제ZIP_위치.md').write_text(f'# 실제 모델·재현 ZIP\n\n`{arc}`\n\nSHA256 `{build["zip_sha256"]}`\n',encoding='utf-8')
    print('DELIVERABLES_READY',n,flush=True)
with (P/'deliverable_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(packages=[10,11],N10_upper=True,all_checks_pass=True,T4_pending=True,E4_access=False,uploaded=False),f,indent=2)
