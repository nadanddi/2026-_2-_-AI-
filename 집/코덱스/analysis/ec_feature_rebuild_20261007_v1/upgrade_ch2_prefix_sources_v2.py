"""Preserve draft v1; freeze metric constants and define numerical fallbacks in v2."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'ch2_prefix_sources_v1.py').read_text(encoding='utf-8')
s=s.replace("WEATHER={'out_temp':1.,'out_hum':1.,'out_wspd':.3,'out_rad':1.}",
            "WEATHER=MappingProxyType({'out_temp':1.,'out_hum':1.,'out_wspd':.3,'out_rad':1.})")
s=s.replace("INDOOR={'in_temp':1.,'in_hum':5.,'in_co2':40.}",
            "INDOOR=MappingProxyType({'in_temp':1.,'in_hum':5.,'in_co2':40.})")
s=s.replace("            assert len(left)==len(right)==3 and days==tuple(sorted(days)) and len(days)==len(set(days))",
            "            assert len(left)==len(right)==len(set(left))==len(set(right))==3\n            assert left==tuple(sorted(left)) and right==tuple(sorted(right))\n            assert days==tuple(sorted(days)) and len(days)==len(set(days))")
start=s.index('    def _distance(');end=s.index('    def _rank(',start)
replacement='''    def _distance(self,index,template,observations):
        block=self.blocks[index];farm=block['farm'];scales=self.scales[farm]
        if any(not finite(value) or value<=0 for value in scales.values()):return None
        terms=[]
        try:
            for h,query in sorted(observations.items()):
                reference=self.records[farm,template][h]
                for c in self.supported_columns[index,h]:
                    if not finite(query[c]):continue
                    delta=query[c]-reference[c]
                    if c in WEATHER:term=WEATHER[c]*(delta/scales[c])**2
                    elif c in INDOOR:term=.25*(delta/INDOOR[c])**2
                    else:term=.25*abs(delta)/50
                    if not finite(term):return None
                    terms.append(term)
            value=math.fsum(terms)/len(terms) if terms else None
        except OverflowError:return None
        return value if finite(value) else None

'''
s=s[:start]+replacement+s[end:]
# Numerical failure in inference is an abstention; it never refits a scale or chooses an arbitrary source.
start=s.index('    def _rank(');end=s.index('    def choose(',start)
old=s[start:end];lines=old.rstrip().splitlines()
s=s[:start]+lines[0]+'\n        try:\n'+'\n'.join('    '+line for line in lines[1:])+'\n        except OverflowError:return None,{\'reason\':\'numerical_signature_overflow\'}\n\n'+s[end:]
compile(s,'ch2_prefix_sources_v2.py','exec')
out=HERE/'ch2_prefix_sources_v2.py';assert not out.exists()
out.write_text(s,encoding='utf-8')
print('CH2 fixed-prefix primitive v2 created; no real-query/model-score execution')
