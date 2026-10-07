from pathlib import Path
out=Path(__file__).parent
source=(out/'report_v1.py').read_text(encoding='utf8')
source=source.replace("'final_verification_v1.json'","'final_verification_v2.json'").replace("'all_file_checks_v1.json'","'all_file_checks_v2.json'").replace("'전수검사_보고서_v1.md'","'전수검사_보고서_v2.md'").replace("'활용가능_외부자료_v1.json'","'활용가능_외부자료_v2.json'")
source=source.replace('개는 모든 행을 읽어 열·결측·값 범위·완전 중복·ID 중복을 검사했다.','개를 전부 확인했고, 5,485개는 모든 행을 읽어 열·결측·값 범위·완전 중복·ID 중복을 검사했다. 나머지 4개는 0바이트의 빈 결과표다.')
source=source.replace('전체 파일 검사 상태:', '검사 코드의 불리언 분위수·NumPy 내부 함수 오류는 새 보완판으로 재검사했다. v1 실패 기록은 보존했다. 최종 남은 읽기 오류 18개는 합성 오류검사 파일 11개, 0바이트 결과 CSV 4개, 주석 포함 소프트웨어 설정 JSON 2개, 실제 0바이트 내용 NPZ 1개로 모두 분류됐다. 대회·외부 원본의 파싱 오류는0개다. 전체 파일 검사 상태:')
source=source.replace('실제 0바이트 내용 NPZ','실제 내용이 전부0인 NPZ')
source=source.replace('이후 새 파일과 변경은 별도 차이 목록으로 남긴다. 원본 해시는 마지막에 재검산했다.','종료 직전 다시 만든 snapshot_delta_v2.json에서 추가·변경·삭제 파일은 모두0개였다. 원본 해시는 마지막에 재검산했다.')
source=source.replace('`final_checks_v1.py`, `final_verification_v1.json`','`final_checks_v2.py`, `final_verification_v2.json`').replace('`inventory_v1.json`, `all_file_checks_v1.json`','`inventory_v1.json`, `all_file_checks_v2.json`')
source=source.replace("'- `sync_rebase_metadata_backup/autostash`", "'- `repair_checks_v2.py`, `repair_log_v2.json`: 불리언 프로필·NumPy 호환 오류 재검사. `issue_triage_v1.json`: 남은 18개 오류의 원인과 근거.',\n'- `snapshot_delta_v2.json`: 전수 검사 전후의 추가·변경·삭제 0개 확인.',\n'- `sync_rebase_metadata_backup/autostash`")
exec(compile(source,str(out/'report_v1.py')+' [v2 corrected checks and issue triage]','exec'))
