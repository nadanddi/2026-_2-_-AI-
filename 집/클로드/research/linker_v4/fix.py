p="feat4.py"; s=open(p,encoding="utf8").read()
old="    gpos={c:np.mean([mp(i) for i in v]) for c,v in by.items()}"
new="""    def cpos(v):
        a=[NUM[i] for i in v if FARM[i]=="F13"]
        return float(np.median(a)) if a else float(np.median([mp(i) for i in v]))   # prefer F13 seg1 numbers
    gpos={c:cpos(v) for c,v in by.items()}"""
assert old in s; s=s.replace(old,new); open(p,"w",encoding="utf8").write(s)
