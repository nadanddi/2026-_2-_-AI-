from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
RUN=Path(__file__).resolve().parent
result=json.loads((RUN/'train_verification_v1.json').read_text(encoding='utf-8'))
assert result['all_checks_pass'] and (RUN/'critic_train_final_v1.json').is_file()
generated=json.loads((RUN/'resource_generated_v2.json').read_text(encoding='utf-8-sig'))
assert generated['total_bytes']==sum(Path(r['path']).stat().st_size for r in generated['files'])==16733799
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
prior=catalog.read_text(encoding='utf-8')
numbers=[int(n) for n in re.findall(r'^\|\s*6\.(\d+)\s*\|',prior,re.M)]
number=max(numbers)+1
line=f'\n| 6.{number} | **이미지 여섯 유형 실제 제작·첫 실자료 학습 작동 시험** (2026-10-10 집 코덱스) | 개발 AIF005/007 원본2장·real2+각양성유형2=14파일, imagegen8호출/실패0. JPEG/마스크/여백/색감 독립PIL 검산 PASS. ImageNet ResNet18/384/AMP batch4/seed20261010/12steps 실제완료,48draws=real24+유형별4. 체크포인트134257747/134259987B SHA와 optimizer/history 재검산, CPU batch/order/reload 최대차이1.19209e-7<1e-5 및 state 불변. 계획/중간/최종 비평 수행; QA1 실행/QA2 등록 참조 불일치 보존 후 재개 전 새wrapper가7핀/QA2 SHA·PASS 실제강제. 중앙경계/색감/단일서비스 지름길 및원본편집이력미인증 한계. 정확도·CV·개선·채택·제출0/팀원0.77참고만/EC중지유지. | 작동시험 통과·성능 미판정 | `집/코덱스/analysis/image_six_type_qa_20261010_v1/REPORT_v2.txt`, `train_verification_v1.json`, `critic_train_final_v1.json` |\n'
with catalog.open('a',encoding='utf-8') as f:f.write(line)
note=f'''\n\n## 2026-10-10 · 집 · 코덱스 여섯 유형 제작 및 첫 실자료 학습 작동 시험 완료
개발 원본415785/833421(금실AIF005·설향AIF007)의 real2+양성6유형각2=14파일 실제 제작. imagegen8호출 성공/재시도0, 서비스시드·모델버전unknown, 생성자산/프롬프트/계보SHA 보존. JPEG90/512/여백 공통, PIL 독립mask/pad/color/codec 검산 PASS. 부분6장은 pre마스크밖0/JPEG밖113~732/16pxhalo밖0;고정중앙경계·색감지름길 미해결.
RTX4050 ResNet18/384/AMPbatch4 첫1step 후11resume=12steps 실제완료/48draws(real24+유형별4),gradient/optimizer계약 통과. CPU 재로드·배치1/3/7·역순·빈입력·state 불변 및 별도checkpoint CPU load batch2/역순5 검산PASS/max1.19209e-7. 첫QA1 gate와등록QA2 hash 참조불일치를중간비평발견;원기록보존/run_finish_v2가재개전7핀·QA2 SHA/PASS강제실행. 계획/중간/최종독립비평 완료, 비평가는metadata/hash/산술검토이며torchload 직접실행아님.
캐시11572412802B<16GiB/이번291562641B<1GiB/기본생성원본별도16733799B/삭제0. 정확도·CV·효용·채택·제출0/원본2장 작동시험범위. 공개21/예약농장이번열람0/팀원v5·0.77참고만/기존EC중지유지. 다음다양한개발농장원본계보분리·56QA·mask/recipe다양화→자체고정검증3seed 비교;train-only체크포인트채택금지. 카탈로그6.{number}. 상세ownanalysis image_six_type_qa_20261010_v1/REPORT_v2.txt,NEXT_STAGE_PLAN_v1.txt. 다른AI작업변경0;동시작업확인없이sync_end자동실행금지.
'''
for path in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-10.md']:
    with path.open('a',encoding='utf-8') as f:f.write(note)
receipt={'catalog_number':f'6.{number}','training_verification_PASS':True,'report_sha256':hashlib.sha256((RUN/'REPORT_v2.txt').read_bytes()).hexdigest(),'default_generated_bytes':16733799,'scope':'stage completion only; no model efficacy claim'}
with (RUN/'completion_record_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
