# 커널 비평 숫자 정정

비평 v1의 합성 Decimal 최대차이를 잘못 적었다. 실제 current receipt 검산 값은 **2.220446049250313e-16**이며 4.440892098500626e-16이 아니다. 384stage/poison10/invalid15 및 두 source SHA MATCH는 그대로다. 산술 순서·범위·필수 caller fullgate 판단은 바뀌지 않는다. 기존 비평을 보존하고 이 새 파일로 정정한다.
