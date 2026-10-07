from pathlib import Path
import sys,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
from PIL import Image
OUT=Path(__file__).parent
path=ROOT/'집/클로드/research/local/deep_cal_6_D.npy';d=np.load(path,allow_pickle=False);mask=np.eye(len(d),dtype=bool)
assert d.shape==(460,460) and np.array_equal(np.isinf(d),mask) and np.isfinite(d[~mask]).all()
distance={'path':path.relative_to(ROOT).as_posix(),'shape':d.shape,'infinite':int(np.isinf(d).sum()),'infinite_only_diagonal':True,'finite_off_diagonal':True,'reason':'distance-matrix self-neighbour exclusion compatible; not an infinite prediction'}
base=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플'
with (base/'sample_labels.csv').open(encoding='utf8-sig',newline='') as f:labels=list(csv.DictReader(f))
images=[]
for r in labels:
    p=base/'images'/r['file']
    with Image.open(p) as im:
        meta={'file':r['file'],'label':r['label'],'type':r['type'],'format':im.format,'size':im.size};im.verify()
    images.append(meta)
assert len(images)==21 and len({r['file'] for r in labels})==21
result={'distance_matrix':distance,'image_mission':{'label_rows':len(labels),'images_verified':len(images),'images':images,'ec_join_key_present':False,'ec_target_present':False,'scope':'separate image classification mission; not EC-labelled observations'}}
p=OUT/'supplementary_checks_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
text=(OUT/'전수검사_보고서_v2.md').read_text(encoding='utf8')
text=text.replace('여러 센서 채널을 합쳐 센 이상 여부는 세부 센서 ID와 함께 판단해야 한다.','여러 센서 채널을 묶어 센 중복의 의미는 세부 센서 ID와 함께 판단해야 한다.')
text=text.replace('무한값 배열은 1개다.','무한값 배열은 1개다. 해당 deep_cal_6_D.npy는 460×460 거리행렬로, 무한값 460개가 대각선에만 있고 대각선 밖은 모두 유한함을 다시 확인했다. 자기 자신을 이웃에서 제외하는 표현과 부합하며 무한 예측값으로 분류하지 않는다.')
text+='\n## 별도 이미지 미션 자료\n\n공개샘플 라벨 21행과 JPG 21개도 파일별로 읽어 이미지 무결성을 확인했다. 이 자료는 이미지 분류 미션으로 배지 EC 목표값이나 정형 row_id 연결 키가 없다. 정형 EC 학습 자료와 별도로 보존하며 12,101개 표·캐시·문서 목록에 이 JPG 21개를 중복 포함하지 않았다. 근거는 supplementary_checks_v1.json이다.\n'
p=OUT/'전수검사_보고서_v3.md'
if p.exists():raise FileExistsError(p)
p.write_text(text,encoding='utf8')
print('SUPPLEMENTARY_PASS images21 diagonal_inf460 REPORT_V3',flush=True)
