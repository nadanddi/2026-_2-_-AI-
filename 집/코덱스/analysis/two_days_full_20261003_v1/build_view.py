from extract import *

def fmt(v):return '결측' if v is None else format(v,'.15g')
def main():
    d=json.loads((HERE/'data.json').read_text(encoding='utf-8'));panels=d['panels'];summary=[];sections=[];md=[]
    title='F13 194일과 F47 191일 전체 비교'
    subtitle='같은 시각끼리 00–23시를 비교했습니다. 차이는 F47−F13입니다. 결측은 0과 구별합니다.'
    md += ['# '+title,'',subtitle,'','원 입력19개 전체, 온도 정답, BASE/CODEX/PFN/W30G 예측을 포함합니다. EC는 기존 공개 검증값만 표시하며 F13 194일은 미열람입니다. 온도예측은 DIAG10 BASE7/CODEX726/PFN문맥1–8 기준입니다.','','| 항목 | 동일 시간 | 다른 시간 | F13 평균 | F47 평균 | 판정 |','|---|---:|---:|---:|---:|---|']
    for p in panels:
        status=p['status'];cls='same' if status=='24시간 완전 동일' else 'missing' if status in ['모두 결측','한쪽 공개값 없음'] else 'different'
        summary.append(f'<tr class="{cls}" data-group="{p["group"]}"><td><a href="#{p["column"]}">{html.escape(p["label"])}</a><small>{p["column"]}</small></td><td>{status}</td><td>{p["same_hours"]}/24</td><td>{p["different_hours"]}/24</td><td>{fmt(p["F13_mean"])}</td><td>{fmt(p["F47_mean"])}</td><td>{fmt(p["max_abs_difference"])}</td></tr>')
        md.append(f'| {p["label"]} ({p["column"]}) | {p["same_hours"]} | {p["different_hours"]} | {fmt(p["F13_mean"])} | {fmt(p["F47_mean"])} | {status} |')
    for p in panels:
        rows=[];md += ['',f'## {p["label"]} — {p["column"]}','',f'{p["status"]}。',{ }.__class__.__name__ if False else '', '| 시각 | F13 194일 | F47 191일 | 差(F47−F13) | 동일 여부 |','|---|---:|---:|---:|---|']
        for r in p['values']:
            flag=r['comparison'];cls='same' if flag=='같음' else 'different' if flag=='다름' else 'missing'
            lv='미열람' if p['column'] in ['sub_ec_public','ec_season_v2_public'] and r['F13'] is None else fmt(r['F13'])
            rv=fmt(r['F47']);delta=fmt(r['difference_F47_minus_F13'])
            if r['difference_F47_minus_F13'] is None:delta='—'
            rows.append(f'<tr class="{cls}"><th>{r["hour"]:02d}:00</th><td>{lv}</td><td>{rv}</td><td>{delta}</td><td>{flag}</td></tr>')
            md.append(f'| {r["hour"]:02d}:00 | {lv} | {rv} | {delta} | {flag} |')
        opening=' open' if p['column']=='in_temp' else ''
        sections.append(f'<details id="{p["column"]}" data-group="{p["group"]}" data-status="{p["status"]}"{opening}><summary>{html.escape(p["label"])} <code>{p["column"]}</code><span>{p["status"]} · 다른 시각 {p["different_hours"]}/24</span></summary><div class="scroll"><table><thead><tr><th>시각</th><th>F13 194일</th><th>F47 191일</th><th>차이 F47−F13</th><th>동일 여부</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')
    page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>두 온실 기록일 전체 비교</title><style>
    :root{color-scheme:light}body{font-family:Segoe UI,Malgun Gothic,sans-serif;background:#f4f6f8;color:#172a3b;margin:0;line-height:1.55}main{max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:28px;margin-bottom:8px}p{margin:8px 0}.note{border-left:4px solid #476780;padding:10px 16px;background:#fff;margin:18px 0}.tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center;position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:10}button,select{padding:9px 12px;border:1px solid #aab8c4;border-radius:6px;background:#fff;color:#172a3b;font-size:14px;cursor:pointer}a{color:#155d90}.scroll{overflow:auto;background:white;border-radius:8px;border:1px solid #d4dde4;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #e3e8ec;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#183b56;color:white}small{display:block;color:#64798b;font-size:11px}.same{background:#eff8f2}.different{background:#fff3ee}.missing{background:#f0f1f3;color:#626a73}details{margin:12px 0;background:white;border:1px solid #d4dde4;border-radius:8px;scroll-margin-top:65px}summary{padding:14px 16px;cursor:pointer;font-weight:600}summary span{float:right;color:#607285;font-weight:400;font-size:13px}details .scroll{margin:0 12px 12px}code{font-size:12px;color:#607285}footer{font-size:13px;color:#586c7d;margin:24px 0}label{font-size:14px}[hidden]{display:none!important}@media(max-width:700px){main{padding:0 12px;margin-top:20px}summary span{display:block;float:none}h1{font-size:23px}}@media print{.tools{display:none}details{break-inside:avoid}main{max-width:none}body{background:white}}
    </style></head><body><main>'''
    page+=f'<h1>{title}</h1><p>{subtitle}</p><div class="note">입력 19개 열 전체와 온도 정답·모델별 예측입니다. EC는 공개 검증 자료만 표시하며 <strong>F13 194일 EC는 미열람</strong>입니다. 연한 초록은 같은 값, 연한 주황은 다른 값, 회색은 결측 또는 미열람입니다. 평균이 같아도 시간별 값이 다르면 동일로 표시하지 않습니다.</div>'
    page+='<div class="tools"><button id="expand">모든 항목 펼치기</button><button id="collapse">접기</button><select id="group"><option value="all">입력·정답·예측 전체</option value="입력">입력만</option><option value="정답">정답만</option><option value="예측·오차">예측·오차만</option></select><label><input id="diff" type="checkbox"> 값이 다른 항목만</label><a href="두날_전체원자료와예측_v1.csv" download>전체 원자료 CSV</a></div>'
    page+='<h2>전체 열 요약</h2><div class="scroll"><table><thead><tr><th>항목</th><th>판정</th><th>같은 시각</th><th>다른 시각</th><th>F13 평균</th><th>F47 평균</th><th>최대 절대차</th></tr></thead><tbody>'+''.join(summary)+'</tbody></table></div><h2>00–23시 모든 값</h2><p>항목을 눌러 펼치거나 위의 ‘모든 항목 펼치기’를 누르세요.</p>'+''.join(sections)
    page+='<footer>온도 예측: DIAG10 W30G, BASE7/CODEX726/PFN문맥1–8. 구동값은 원 단위를 보존했습니다. 원자료 CSV는 원 수치 정밀도를 보존하며, 정답이나 예측을 입력과 구분했습니다. 결측 5개 열은 두 날 모두 24시간 결측입니다.</footer></main><script>const ds=[...document.querySelectorAll("details")];document.getElementById("expand").onclick=()=>ds.filter(d=>!d.hidden).forEach(d=>d.open=true);document.getElementById("collapse").onclick=()=>ds.forEach(d=>d.open=false);function filter(){const g=document.getElementById("group").value;const only=document.getElementById("diff").checked;ds.forEach(d=>d.hidden=(g!=="all"&&d.dataset.group!==g)||(only&&!d.dataset.status.includes("다름")));document.querySelectorAll("tbody tr[data-group]").forEach(r=>r.hidden=(g!=="all"&&r.dataset.group!==g)||(only&&!r.classList.contains("different")));}document.getElementById("group").onchange=filter;document.getElementById("diff").onchange=filter;document.querySelectorAll("a[href^=\"#\"]").forEach(a=>a.onclick=()=>{const d=document.querySelector(a.getAttribute("href"));if(d){d.hidden=false;d.open=true;}});</script></body></html>'
    (HERE/'전체비교표_v1.html').write_text(page,encoding='utf-8')
    (HERE/'전체비교표_v1.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'27항목×24시간={27*24}대조값, 전체 HTML/Markdown 생성')

if __name__=='__main__':main()
