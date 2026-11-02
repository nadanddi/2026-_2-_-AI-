from pathlib import Path
p=Path(__file__).parent
s=(p/'run_v2.py').read_text(encoding='utf-8')
s=s.replace("OUT=Path(__file__).resolve().parent", "OUT=Path(__file__).resolve().parent/'corrected_v3';OUT.mkdir(exist_ok=True)")
s=s.replace("ss,gg=lineage(held);return np.array([norm(r) not in ss and r.get('group') not in gg for r in rr])", "ss,gg=lineage(held);return np.array([norm(r) not in ss for r in rr])")
s=s.replace("keep=(ds!=held)&disjoint(rr,hr);# keep other batches but purge overlapping original/group", "keep=(ds!=held)&(~np.isin(ds,NEG)|disjoint(rr,hr));# original request: other three positives, exclude paired real only")
s=s.replace("overlap=int(((ds!=held)&np.isin(ds,POS)&~disjoint(rr,hr)).sum())", "overlap=int(((ds!=held)&np.isin(ds,POS)&~disjoint(rr,hr)).sum())")
s=s.replace("removed_other_positive_overlap=overlap", "retained_other_positive_source_overlap=overlap")
s=s.replace("   while not (OUT/'MIDPOINT_APPROVED.txt').exists():\n    assert datetime.now(timezone(timedelta(hours=9))).hour<15,'deadline reached during review'\n    time.sleep(2)", "   while not (OUT.parent/'CORRECTION_APPROVED.txt').exists():\n    assert datetime.now(timezone(timedelta(hours=9))).hour<15,'deadline reached during review'\n    time.sleep(2)")
(p/'run_v3.py').write_text(s,encoding='utf-8')
s=(p/'verify_v1.py').read_text(encoding='utf-8').replace('import run_v2 as R','import run_v3 as R').replace("OUT=Path(__file__).resolve().parent", "OUT=Path(__file__).resolve().parent/'corrected_v3'").replace("sys.path.insert(0,str(OUT))", "sys.path.insert(0,str(OUT.parent))")
(p/'verify_v2.py').write_text(s,encoding='utf-8')
s=(p/'run_remote_v2.sh').read_text(encoding='utf-8').replace('run_v2.py','run_v3.py')
(p/'run_remote_v3.sh').write_text(s,encoding='utf-8',newline='\n')
