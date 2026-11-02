import zipfile, json, io, hashlib, os, numpy as np
src=r'연구실/클로드/local/submissions_img/팜모니_나단9.zip'; dst=r'연구실/클로드/local/submissions_img/팜모니_나단11.zip'
E=r'연구실/코덱스/local/im_egpt_20261016_v1/corrected_v3/'
zi=zipfile.ZipFile(src); P='팜모니_나단9/'; Q='팜모니_나단11/'
inf=zi.read(P+'inference.py').decode('utf-8').replace('\r\n','\n')
a='    gz = np.load(model_dir / "gr_expert.npz"); hx["E_gr"] = {a: gz[f"E_gr__{a}"] for a in ("mean", "scale", "coef", "intercept")}\n'
assert inf.count(a)==1
inf=inf.replace(a, a+'    xz = np.load(model_dir / "gpt_expert.npz"); hx["E_gpt"] = {a: xz[a] for a in ("mean", "scale", "coef", "intercept")}   # 나단11\n')
b='''        except Exception:
            pass
    return min(max(p, 0.0), 1.0)
'''
assert inf.count(b)==1
inf=inf.replace(b,'''        except Exception:
            pass
    try:   # 7) 나단11: GPT 편집 전문가 E_gpt — 형식 게이트 안이고 진짜 확정(rc)이 아닐 때만, p ← max(p, g(s, θ_gpt))
        if info.get("std", False) and not rc:
            s = float(_lr(m["hx"]["E_gpt"], np.r_[low, info["emb"]][None])[0])
            if np.isfinite(s): p = max(p, _g(s, m["cfg"]["gpt"]["theta"]))
    except Exception:
        pass
    return min(max(p, 0.0), 1.0)
''')
inf='"""팜모니 이미지 미션 제출 11 — 나단9 + GPT 편집 전문가 E_gpt(저수준 38 + L/14 조각평균 임베딩 로지스틱, model/gpt_expert.npz).\n 7) 형식 게이트 안이고 정밀 원본 대조의 진짜 확정(rc)이 아닐 때만 p ← max(p, g(s_gpt, θ_gpt)). θ_gpt는 model/config.json "gpt".\n(이하 제출 09 설명) '+inf[3:]
compile(inf,'inference.py','exec')
cfg=json.loads(zi.read(P+'model/config.json')); th=json.load(open(E+'final_theta.json'))
cfg["version"]="nadan11 = nadan9 + E_gpt(rc 제외)"; cfg["gpt"]={"theta":th["theta"],"note":"E_gpt: 양성 GPT 편집·생성 4묶음(gpt1·gpt2·gpt3·gptr), 음성 h3 진짜+흔들기 진짜+gptr 짝 진짜. θ=진짜 0.25% 분위. 진짜 확정 사진엔 미적용(GPT 가짜 rc 0/600)."}
fm=np.load(E+'final_model.npz'); b2=io.BytesIO(); np.savez(b2, **{k:fm[k] for k in ("mean","scale","coef","intercept")})
readme=open(os.path.join(os.path.dirname(__file__),'readme11.md'),encoding='utf-8').read()+'\n(아래는 나단9 설명)\n\n'
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as zo:
    for i in zi.infolist():
        nm=i.filename.replace(P,Q,1); b=zi.read(i.filename)
        if i.filename==P+'inference.py': b=inf.replace('\n','\r\n').encode('utf-8')
        elif i.filename==P+'model/config.json': b=json.dumps(cfg,ensure_ascii=False,indent=1).encode('utf-8')
        elif i.filename==P+'README.md': b=readme.encode('utf-8')+b
        zo.writestr(zipfile.ZipInfo(nm,date_time=i.date_time), b, compress_type=zipfile.ZIP_DEFLATED)
    zo.writestr(zipfile.ZipInfo(Q+'model/gpt_expert.npz',date_time=(2026,10,16,14,20,0)), b2.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
print(os.path.getsize(dst), hashlib.sha256(open(dst,'rb').read()).hexdigest())
