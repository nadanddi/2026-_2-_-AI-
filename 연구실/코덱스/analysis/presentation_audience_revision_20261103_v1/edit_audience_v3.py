"""Edit a new copy of G05v4, preserving source/package structure and numeric data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from io import BytesIO
import json, re, hashlib
from lxml import etree as E
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[4]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/'연구실/클로드/발표/덱/팜모니AI_발표자료_G05v4.pptx'
TARGET=WORK/'팜모니AI_발표자료_청자이해보완_v3.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
A=NS['a']; changes=[]

EXACT={
1:{'정형: 시설 딸기 배지 온도·EC 예측   /   이미지: 딸기 사진의 실제·가짜 판별':'센서 표 자료: 배지 온도·EC 추정   /   사진: 실제·생성·편집 판별'},
2:{'최종 답안: 정형 온도 0.4608·EC 0.1148, 이미지 Accuracy 0.97':'배지 온도 오차 0.4608℃, EC 오차 0.1148 dS/m, 사진 정답률 97%',
'온도 RMSE(℃)':'온도 오차 RMSE(℃)','EC RMSE':'EC 오차 RMSE(dS/m)','Accuracy':'정답률(Accuracy)',
'표 0. 최종 검증에 제출한 코드·모델의 점수. RMSE는 낮을수록 좋음':'표 0. 대회 공식 성적. RMSE = 큰 오차에 더 민감한 오차 지표(낮을수록 좋음). 정답률은 높을수록 좋음.'},
3:{'두 온실(F13, F47)의 시간별 배지 온도와 EC 예측':'배지 = 뿌리를 심는 재료. 온도와 전기전도도(EC)를 시간별 추정'},
4:{'기록이 섞여 있고, 하루가 두 온실 ID를 넘나들며 이어진다':'서로 다른 ID의 하루 기록 사이에서 연속성을 발견했다',
'온실':'제공된 온실 ID','400일, 9,600행':'하루 기록 400개, 9,600행','60기록(모두 2차 구간)':'하루 기록 60개(채점 대상)',
"오늘(4일)의 '어제'는 다른 ID에 있을 수 있다.":'같은 동의 과거 기록이 다른 ID에 담겼을 수 있다.',
'그림 1. 개념도(날짜는 예시)':'그림 1. 동일한 물리 동의 이력을 복원한다는 해석(날짜는 예시)'},
5:{"하루치 오차의 답은 '어제'에 있다":'하루 전체가 높거나 낮게 빗나가는 오차에 주목했다',
'오차 중 하루 통째 몫':'하루 평균 어긋남의 몫','그날 입력으로 맞힌 정도(R²)':'그날 입력의 설명력(R²)','어제 오차와 닮은 정도(상관)':'전날·당일 평균 오차의 상관',
'F13·F47':'예측 대상 두 ID','F10·F11':'중복 기록 온실','F32':'배지가 더 따뜻한 온실',
"그래서 같은 동의 진짜 '어제'를 찾아 잇는다.":'전날 상태가 이어진다는 관찰에서, 과거 실측·예측을 쓰는 방법을 시험했다.',
'표 3. 데이터를 현장 상황에 빗대면(학습 자료 온실 51곳)':'표 3. 자료 품질 진단. 온실별 상세 ID·해석 한계는 노트'},
6:{'세 모델 섞기\n8℃ 이하는 TabPFN 제외':'세 예측 모델의 평균\n8℃ 이하는 TabPFN 제외',
'어제 실측 → 앵커\n실측 없음 → 사슬':'어제 측정값이 있으면 보정\n없으면 앞선 예측을 이어받음',
'3단계  추운 행 보정':'3단계  추운 시간 보정','고EC 날 추운 시간대\n+ 소폭 보정(TX3)':'EC가 높은 날의\n추운 시간에 소폭 보정',
'그림 2. 온도 예측의 세 단계(EC는 기본 예측 + 어제 EC 앵커)':'그림 2. 온도 모델의 역할 요약. EC는 별도 기본 모델·하루 보정·어제 실측 보정.',
'그림 3. 어제 23시 값의 반영 비율(β_h)\n가로축: 오늘의 시각(0~23시)\n세로축: 반영 비율(0.2 = 차이의 20%를 더함)\n새벽엔 어제가, 낮엔 오늘 입력이 주도':'그림 3. 어제 실측과 오늘 기본 예측의 차이를 반영하는 비율\n가로축: 오늘의 시각(0~23시)\n세로축: 보정 비율(0.2 = 차이의 20% 반영)\n오늘 정답·오늘 오차를 입력으로 쓰는 것은 아님'},
7:{"대회 자료에만 있는 표식은 빼고, 현장에도 있는 '어제'만 남겼다":'현장에서 확보할 정보와 대회 전용 규칙을 구분했다',
'번호 표식, 평가 유무 규칙':'번호 구간·평가 구성 규칙','같은 동의 어제':'같은 동의 전날 이력','달력으로 정해짐':'날짜·동 식별자로 확인',
'계절 지표 날짜 복원':'외기로 계절·날짜 추정','미실시':'달력 대체 성능 미확인'},
8:{'기준을 통과한 변경은 실제로도 좋아졌고, 못 넘은 변경 두 건은 나빠졌다':'여러 조건에서 검증하고, 대회 점수 변화와 비교했다',
'사전 기준':'내부 검증 판정','현장형 모델':'구조 규칙 제거','이력 다시 잇기':'과거 기록 재연결','사슬 문턱 0.8 → 0.5':'전날 연결 허용 확대','앵커·사슬 확장(최종)':'과거 상태 보정 확대(최종)',
'제출 비교':'공개 점수로 선택','범위 밖 보정 추가':'학습 범위 밖 온도 보정',
'표 5. 회차별 사전 기준 판정(기준은 노트)':'표 5. 실험 전 정한 내부 기준의 판정. 공개 점수로 고른 변경도 있음.'},
9:{"'어제' 정보를 끄면 오차가 온도 3~10%, EC 8~31% 늘어난다":'과거 상태 보정을 빼면 온도 오차 3~10%, EC 오차 8~31% 증가',
"그림 5. '어제'를 끄고 일반 방법만 남겼을 때의 오차 증가(%). 가로축은 검증기(학습 기록을 가려 만듦).":'그림 5. 최종 모델 대비 오차의 상대 증가율(%). 가로축은 학습 기록을 숨기는 검증 방식. 계절 특징은 유지.',
'현장 조건: 센서 기록이 끊기지 않고, 어제 실측이 남아 있어야 한다.':'실측 보정은 전날 측정값이 있을 때 사용. 온도는 앞선 예측을 이어받는 방법도 사용.'},
10:{'딸기 시설재배 사진이 실제인지, 생성·편집된 가짜인지 판별':'딸기 사진의 실제 촬영본과 생성·편집본을 구분하는 별도 과제'},
11:{'정답을 아는 학습 자료를, 공개 데이터로 우리가 직접 만들었다':'학습 사진이 제공되지 않아, 실제 사진으로 가짜를 직접 만들었다',
'AI Hub 딸기\n152,043장':'공개 딸기 사진 원천\n152,043장',
'필터·합성\nSD·SDXL 생성\nQwen·ChatGPT 편집':'필터·합성·AI 생성\n원본 일부·전체 편집\n제작 도구는 노트',
'768×1024\nJPEG Q92, 4:2:0':'크기·압축 조건 통일\n768×1024, JPEG\n세부 설정은 노트',
'묶음 단위 교차검증\n봉인 묶음':'자료를 나눠 학습·검증\n検証写真'.replace('検証写真','검증용 사진 별도 확보'),
'그림 6. 학습 자료를 만든 순서(가짜 종류는 노트)':'그림 6. 원천 전체를 모두 학습한 것은 아님. 기본 판별기: 실제 2,467장 + 가짜 7,124장.'},
12:{'핵심은 사진 한 장만 보는 일반 판별기다':'사진의 내용·처리 흔적을 보고, 놓치는 유형을 추가 검사한다',
'CLIP 의미 특징\n저수준 통계 38개\n→ 두 확률의 평균':'사진 내용 특징(CLIP)\n잡음·선명도 등 38특징\n두 판별 결과의 평균',
'유형별 전문가':'유형별 추가 판별기','부분 생성\n부분 스타일\n국소 조각':'일부 영역 생성\n질감·색감 편집\n작은 영역의 변형',
'대회 자료용 장치':'원본·저장 형식 보조',
'지문 DB 원본 대조: AI Hub 원본이 있을 때\n형식 게이트: JPEG Q92, 4:2:0일 때':'원본 요약 DB: 등록된 원본과 차이 비교\n저장 형식 조건: 맞을 때 일부 추가 판별기 사용',
'그림 7. 판별 순서. 점선은 대회 자료에서만 작동하고 현장 사진에서는 거의 꺼지는 장치.':'그림 7. 등록 원본이 없으면 대조를 생략하고, 저장 형식이 다르면 일부 판별기를 생략한다.'},
13:{'학습에 쓰지 않은 검증 묶음에서 균형 정확도 0.94~0.96':'학습에 넣지 않은 사진에서 균형 정확도 94~96%',
'묶음':'사진 묶음','원본 겹침':'학습과 같은 원본','그림 8. 봉인 묶음의 균형 정확도(%)':'그림 8. 실제·가짜 각각의 정답률 평균(%). 높을수록 좋음.',
'표 6. 검증 묶음(사진은 학습에 안 씀). 대표값은 R3~R5':'표 6. 검증 1~3은 같은 원본 없음. 농장·촬영 묶음은 학습과 겹침.',
'Acc 0.97':'정답률 0.97'},
14:{'대회 자료용 장치를 모두 끄면 균형 정확도가 2.7~3.0%p 내려간다':'원본 대조·형식 조건을 끄면 정확도가 하락하고, 사진 처리에도 취약하다',
'그림 9. 장치를 끈 균형 정확도(%). 둘 다 끔 = 지문 DB와 형식 게이트를 모두 끈 판(현장 사진에 가까운 조건).':'그림 9. 균형 정확도(높을수록 좋음). 두 보조 처리를 끄면 2.7~3.0%p 하락. 같은 AI Hub 출처의 비교.',
'그림 10. 실제 사진을 다시 저장·처리했을 때 가짜로 잘못 판정한 비율(%). R3 실제 429장.':'그림 10. 제출 모델이 진짜 사진을 가짜로 본 비율(낮을수록 좋음). 검증 1의 실제 429장을 처리.',
'현장에서는 그 농장 사진으로 같은 절차를 다시 돌려 확인해야 한다.':'현재는 휴대폰 일반 사진 성능을 보장할 수 없음. 현장 자료로 재학습·재검증이 필요.'},
15:{'예측·판정 단계에는 AI 도우미를 쓰지 않는다.':'실제 예측은 저장 모델로 실행하며, Claude·ChatGPT에 원격으로 묻지 않는다.',
'기준 사전 고정':'실험 전에 채택 기준 결정','규정 경계':'허용 정보 범위'},
16:{'한 것과 아직 하지 않은 것을 나눠 적는다':'과거 상태 활용과 사진 판별을 구현했고, 현장 검증은 남아 있다',
'어제를 이어 하루 오차를 줄임\n대회 표식을 뺀 구조':'과거 상태 보정 효과 확인\n대회 구성 규칙 제거',
'달력으로 계절 지표 교체\n실제 어제로 재검증':'실제 날짜·전날 정보로\n최종 모델 재검증',
'가짜를 만들어 학습·검증\n장치를 끈 성능 측정':'자체 제작 자료로 학습·검증\n보조 처리를 끈 성능 확인'},
23:{'다른 출처 자료: 사슬 식을 다른 49개 온실에서 확인, 48곳 개선':'같은 대회 자료의 다른 49개 온실: 전날 예측 이어받기 식으로 48곳 개선',
'어제 찾기 링커':'전날 이력 복원·보정','달력으로 어제가 정해짐':'날짜·동 식별자로 확인','끈 성능':'과거 보정 제거 영향'},
30:{'현장 사진은 대부분 두 장치 모두 작동하지 않는다.':'현장 활성 비율은 미측정. 등록 원본·저장 형식에 따라 적용 여부가 달라진다.'},
33:{'백본':'사진 특징 추출 모델','가짜 5판 포함, 최종 나단11':'추가 가짜 포함, 최종 제출판 선택'}
}

SPOKEN={
1:'저희는 딸기 온실의 센서 표 자료와 사진, 두 과제를 풀었습니다. 첫째는 다른 센서로 배지 온도와 EC를 추정하는 일이고, 둘째는 실제 촬영 사진과 생성·편집 사진을 구분하는 일입니다. 공통으로 현장에서 확보할 수 있는 정보와 대회 자료에만 있는 조건을 구분했습니다.',
2:'공식 성적은 배지 온도 오차 0.4608도, EC 오차 0.1148 dS/m, 사진 정답률 97%입니다. 온도와 EC는 큰 오차에 더 민감한 RMSE로, 낮을수록 좋습니다. 이 수치가 매번 그 오차 이내라는 뜻은 아닙니다. 사진은 맞힌 비율입니다. 뒤에서 어떻게 검증했고 어디까지 현장에 적용할 수 있는지 설명하겠습니다.',
3:'배지는 딸기 뿌리를 심는 재료입니다. EC는 전기전도도로, 양분·염류 상태를 파악하는 데 쓰는 지표입니다. 여기서는 실내 공기 온도를 예측하는 것이 아니라, 같은 시간대의 다른 센서와 구동기 기록으로 배지 상태를 추정합니다.',
4:'학습은 하루 24시간짜리 기록 400개, 채점 대상은 60개입니다. 실제 날짜와 온실 세부 정보는 제공되지 않았습니다. F13과 F47은 제공된 ID입니다. 그런데 학습 정답의 온도·EC와 외기가 함께 이어지는 고확신 짝 190개 중 40개가 ID를 넘었습니다. 같은 물리 동의 과거 기록이 여러 ID에 분산된 것으로 해석했지만, 원래 동 식별자와 오류 원인은 확인하지 못했습니다. 오른쪽은 그 복원 개념도입니다.',
5:'먼저 하루 안의 곡선 모양보다 하루 전체가 높거나 낮게 어긋나는 오차를 봤습니다. 예를 들어 하루 내내 1도 높게 예측하는 경우입니다. 다른 49개 온실에서 이 하루 수준 오차가 제곱오차의 약 절반을 차지했고, 전날과 당일의 평균 오차가 비슷했습니다. 그날 입력 요약만으로 설명하는 시험은 성능이 낮았습니다. 이것은 관찰 결과이며 오늘 정답을 예측에 넣었다는 뜻은 아닙니다. 그래서 전날 상태를 활용하는 방법을 시험했습니다.',
6:'온도는 세 모델의 예측을 섞고, 추위에 약했던 사전학습 모델 TabPFN의 비중을 줄였습니다. 과거 기록을 찾는 절차를 링커라고 부릅니다. 어제 실측과 오늘 기본 예측의 차이로 보정하는 것이 앵커입니다. 어제 실측이 없으면 조건에 맞는 앞선 예측을 이어받는 사슬을 씁니다. 그래프는 그 보정 차이를 오늘 각 시간에 얼마나 반영하는지입니다. 오늘 정답을 안다는 뜻은 아닙니다. EC는 별도 기본 모델과 하루 보정, 어제 실측 보정을 사용합니다.',
7:'여기서는 일반적인 날짜 이력과 대회 전용 규칙을 구분했습니다. 번호 구간이나 평가 기록 유무에 따른 규칙은 제거했습니다. 같은 동의 과거 센서 이력은 남겼습니다. 대회에서는 날짜·동 정보가 부족해 이를 복원하지만, 정상적인 현장 로그에서는 날짜와 동 식별자로 확인할 수 있습니다. 외기로 계절을 복원하는 부분은 남아 있고, 실제 달력으로 대체한 최종 성능은 아직 재지 않았습니다.',
8:'학습 자료 중 일부 기록의 정답을 숨기고 맞혀 보는 방식으로 검증했습니다. 추운 날을 숨기는 방식과 무작위로 나누는 방식을 함께 썼습니다. 반복 학습과 여러 검증 조건에서 개선 방향을 확인하고 불확실성도 비교했습니다. 왼쪽은 대회 제출 순서에 따른 온도 오차로 낮을수록 좋습니다. 내부 기준 통과와 실제 공개 점수의 변화가 일치한 사례가 있었지만, 사례 수가 작습니다. 기준을 모두 통과하지 못하고 공개 점수로 선택한 변경도 따로 밝혔습니다.',
9:'이 그림은 같은 검증에서 과거 상태 보정을 제거했을 때의 상대 오차 증가입니다. 추운 날 검증에서는 온도 오차가 약 10% 늘었고, EC는 검증 방식에 따라 약 8~31% 늘었습니다. 최종 모델의 오차를 기준으로 한 증가율이며 계절 특징은 양쪽에 남아 있습니다. 어제까지 측정됐지만 오늘 배지 센서가 누락된 경우에 실측 보정을 활용할 수 있습니다. 온도는 실측 대신 앞선 예측을 이어받기도 합니다. 실제 날짜 정보로 대체한 최종 전체 모델의 현장 검증은 남아 있습니다.',
10:'두 번째는 사진 과제입니다. 실제 촬영 사진과 생성하거나 편집한 사진을 구분합니다. 센서 예측과 연결된 하나의 서비스는 아니며 별도로 수행한 대회 과제입니다.',
11:'정답이 붙은 학습 사진이 제공되지 않아 공개 딸기 사진에서 가짜를 직접 만들었습니다. 일부 영역을 바꾸거나 전체를 생성하는 등 여러 방법을 썼습니다. 15만 장은 원천 사진 전체 수이고 모두 기본 판별기 학습에 넣은 것은 아닙니다. 기본 판별기에는 실제 2,467장과 가짜 7,124장을 썼습니다. 실제와 가짜 모두 크기·압축 조건을 맞췄고 검증용 사진을 따로 확보했습니다. 조건을 맞춘 것만으로 편향이 없다고 보장되지는 않습니다.',
12:'기본 판별기는 사진 내용의 특징과 잡음·선명도 같은 처리 흔적을 함께 봅니다. 추가 판별기는 일부 생성이나 질감 편집처럼 놓치기 쉬운 유형을 봅니다. 점선 안은 두 보조 처리입니다. 원본 요약 DB는 등록 원본과 차이를 비교하고, 저장 형식 조건은 해당 형식에서 학습한 일부 판별기만 켭니다. 등록 원본이 없거나 형식이 다르면 해당 처리를 생략합니다. 전문가별 오탐 예산은 학습 자료에서 정한 목표이며 전체 현장 오탐률 보장은 아닙니다.',
13:'검증 1·2·3은 학습에 사진 자체를 넣지 않은 세 묶음입니다. 균형 정확도는 실제 사진과 가짜 사진 각각의 정답률 평균으로 94~96%였습니다. 참고 묶음은 일부 실제 사진의 원본으로 만든 가짜가 학습에 있어 대표값에서 제외했습니다. 검증 묶음은 후보 선택에도 사용했고 농장·촬영 묶음도 학습과 겹칩니다. 따라서 새 농장 성능은 아닙니다. 공식 정답률 97%는 다른 시험과 지표여서 직접 비교하지 않습니다.',
14:'왼쪽은 같은 출처의 검증 사진으로 원본 대조와 저장 형식 조건을 생략한 비교입니다. 높을수록 좋고, 두 처리를 모두 끄면 2.7~3.0퍼센트포인트 내려갑니다. 오른쪽은 다른 비교입니다. 제출 모델에 진짜 사진을 처리해 넣었을 때 가짜로 잘못 보는 비율로 낮을수록 좋습니다. 선명화 90.7%, WebP 저장 37.5%로 취약했습니다. 같은 사진을 저장만 바꿔도 오판할 수 있다는 뜻입니다. 현재 일반 휴대폰 사진 성능을 보장하지 못하며 현장 자료로 재검증해야 합니다.',
15:'팀은 문제 해석과 채택 기준, 최종 선택을 맡았고 AI 도구는 제안·코드·실험·초안을 도왔습니다. ChatGPT로 학습용 가짜 사진도 만들었습니다. 실제 예측은 저장된 모델로 실행합니다. 사전학습 AI 모델은 쓰지만, 매번 Claude나 ChatGPT에 원격으로 물어보는 방식은 아닙니다.',
16:'정형에서는 전날 상태를 활용하는 보정의 효과를 확인했습니다. 다만 정상적인 날짜·동 정보로 대체한 최종 모델 검증은 남아 있습니다. 이미지는 학습 자료를 직접 만들고 보조 처리를 끈 성능까지 확인했지만, 사진 저장·보정 변화와 새 농장에서는 검증이 더 필요합니다. 구현한 결과와 앞으로 확인할 조건을 구분해 말씀드렸습니다. 감사합니다.'}

def image_label(s):
    # Match whole internal identifiers, not the official scoring round R21.
    names={'R2':'참고 묶음','R3':'검증 1','R4':'검증 2','R5':'검증 3'}
    s=s.replace('R3~R5','검증 1~3').replace('R2~R5','참고·검증 묶음')
    return re.sub(r'(?<![A-Za-z0-9_])R[2345](?![A-Za-z0-9_])',lambda m:names[m.group()],s)

GENERAL=[('형식 게이트','저장 형식 조건'),('지문 DB','원본 요약 DB'),('링커','전날 찾기'),('HR5','EC 하루 보정'),('TX3','추위 보정'),('봉인 묶음','검증 사진 묶음'),('봉인 검증','사진 검증')]
def plain(s,n):
    s=EXACT.get(n,{}).get(s,s)
    if n in [13,14,24,25,31]:s=image_label(s)
    for a,b in GENERAL:s=s.replace(a,b)
    s=s.replace('사슬','예측 이어받기').replace('앵커','실측 보정').replace('예측 예측 이어받기','예측 이어받기')
    if n==25 and s.startswith('[정정]'):s='[정정] 기존 README의 촬영 묶음 비중복 설명을 정정합니다. 검증 사진은 학습에 넣지 않았지만 농장·촬영 묶음은 모두 겹칩니다. 같은 원본 겹침이 없는 검증 1~3을 대표값으로 사용합니다.'
    return s

def rewrite_body(body,new):
    oldps=body.findall(f'{{{A}}}p'); template=deepcopy(oldps[0])
    for p in oldps:body.remove(p)
    for line in new.split('\n'):
        p=deepcopy(template)
        runs=p.findall(f'{{{A}}}r')
        proto=deepcopy(runs[0]) if runs else E.Element(f'{{{A}}}r')
        for c in list(p):
            if c.tag not in [f'{{{A}}}pPr',f'{{{A}}}endParaRPr']:p.remove(c)
        for c in list(proto):
            if c.tag!=f'{{{A}}}rPr':proto.remove(c)
        E.SubElement(proto,f'{{{A}}}t').text=line
        end=p.find(f'{{{A}}}endParaRPr')
        if end is None:p.append(proto)
        else:p.insert(list(p).index(end),proto)
        body.append(p)

def chart_label(s):
    s=image_label(s)
    for a,b in [('EXT10','추운 날 10℃'),('EXT12','추운 날 12℃'),('DIAG10','무작위 분할'),('P2LOO','후반 1일씩'),('EL1','후반 기록 묶음'),('지문 DB 끔','원본 대조 끔'),('둘 다 끔','두 처리 끔'),('4:4:4 저장','색 정보 저장 변경')]:s=s.replace(a,b)
    return s

assert not TARGET.exists()
with ZipFile(SOURCE) as src, ZipFile(TARGET,'w',ZIP_DEFLATED) as dst:
    for info in src.infolist():
        data=src.read(info.filename); m=re.fullmatch(r'ppt/slides/slide(\d+)\.xml',info.filename)
        if m:
            n=int(m.group(1)); root=E.fromstring(data)
            for body in root.findall('.//p:txBody',NS)+root.findall('.//a:txBody',NS):
                vals=[t.text or '' for t in body.findall('.//a:t',NS)]; old='\n'.join(vals)
                new=plain(old,n)
                if new!=old:
                    rewrite_body(body,new);changes.append({'slide':n,'before':old,'after':new})
            data=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        elif re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',info.filename):
            n=int(re.search(r'notesSlide(\d+)',info.filename).group(1));root=E.fromstring(data)
            bodies=root.findall('.//p:sp',NS)
            for sp in bodies:
                ph=sp.find('.//p:ph',NS)
                if ph is None or ph.get('type')!='body':continue
                body=sp.find('p:txBody',NS)
                if body is None:continue
                old='\n'.join(t.text or '' for t in body.findall('.//a:t',NS))
                original=old.split('[노트]',1)[1] if '[노트]' in old else old
                if n in SPOKEN:new='[말할 것]\n'+SPOKEN[n]+'\n\n[근거·검토 노트]\n'+original
                else:new=old
                # Preserve original evidence/source names, but identify ambiguous terms.
                new+='\n\n[청자용 용어 대응]\n검증 1 = 기존 R3(실제 429/가짜 471), 검증 2 = 기존 R4(355/345), 검증 3 = 기존 R5(356/344). 참고 묶음 = 기존 R2(1,862/1,862; 동일 원본 파생본 겹침 있음). 검증 1~3도 농장·촬영 묶음은 학습과 겹치며 후보 선택에 사용됨.\n전날 찾기 = 링커, 실측 보정 = 앵커, 예측 이어받기 = 사슬. EC 하루 보정 = HR5, 추위 보정 = TX3. 원본 요약 DB = fp_db/finefp_db, 저장 형식 조건 = 형식 게이트.\n추운 날 10℃/12℃ = EXT10/EXT12(해당 하루 최저온도 미만 기록을 가리는 검증), 무작위 분할 = DIAG10, 후반 기록 묶음 = EL1, 후반 1일씩 = P2LOO.\n화면의 쉬운 이름은 표시 변경이며 모델·계수·학습·검증 수치를 바꾸지 않음.'
                if n==4:new+='\n[추가 확인] 190쌍 중 ID 간 40쌍은 원본 CSV로 독립 재현. 실제 물리 동·오류 원인은 미확인. 기존 128개 날짜 그룹·4개 동 설명은 집계 기준 의존 해석이며 실제 날짜·물리 동 확정값으로 사용하지 않음. 같은 폴더의 현장오류와_대회구조_근거확인_v2.md 참조.'
                if n==23:new+='\n[범위 보충] 위 제거 수치는 앵커·사슬 제거이며 정상 메타데이터로 링커만 대체한 성능이 아님. PCH4는 같은 대회 자료의 다른 49개 온실에서 사슬 식 48곳 개선(카탈로그 6.983); 최종 전체 모델·새 출처 성능은 아님.'
                rewrite_body(body,new)
            data=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        elif re.fullmatch(r'ppt/charts/chart\d+\.xml',info.filename):
            root=E.fromstring(data)
            for node in root.findall('.//c:strCache/c:pt/c:v',NS)+root.findall('.//a:t',NS):
                if node.text:node.text=chart_label(node.text)
            data=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        elif info.filename.startswith('ppt/embeddings/') and info.filename.endswith('.xlsx'):
            buf=BytesIO()
            with ZipFile(BytesIO(data)) as zi, ZipFile(buf,'w',ZIP_DEFLATED) as zo:
                for xi in zi.infolist():
                    xb=zi.read(xi.filename)
                    if xi.filename.endswith('.xml'):
                        xr=E.fromstring(xb)
                        for t in xr.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                            if t.text:t.text=chart_label(t.text)
                        xb=E.tostring(xr,encoding='UTF-8',xml_declaration=True,standalone=True)
                    zo.writestr(xi,xb)
            data=buf.getvalue()
        dst.writestr(info,data)

# Source and chart numeric caches remain unchanged; native slide/table/chart objects retained.
with ZipFile(SOURCE) as a,ZipFile(TARGET) as b:
    for name in a.namelist():
        if re.fullmatch(r'ppt/charts/chart\d+\.xml',name):
            def nums(z):return [e.text for e in E.fromstring(z.read(name)).findall('.//c:numCache/c:pt/c:v',NS)]
            assert nums(a)==nums(b),name
    assert len([x for x in b.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',x)])==33
    assert len([x for x in b.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',x)])==33
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='c24c4bc0eff00c80a160d089e5dd4555f42bc078bba050e75164d3db292c9a24'
with (WORK/'변경목록_v3.json').open('x',encoding='utf-8') as f:json.dump(changes,f,ensure_ascii=False,indent=2)
with (WORK/'발표대본_청자이해보완_v2.md').open('x',encoding='utf-8') as f:
    f.write('# 발표 대본 — 청자 이해 보완 v1\n\n수정 PPTX 1~16장 노트의 [말할 것]과 같은 대본이다. 실제 리허설 시간은 미측정.\n\n')
    for n,s in SPOKEN.items():f.write(f'## {n}장\n\n{s}\n\n')
print(json.dumps({'pptx':str(TARGET),'slide_text_changes':len(changes),'slides':33,'notes':33,'chart_numbers_unchanged':True},ensure_ascii=False))
