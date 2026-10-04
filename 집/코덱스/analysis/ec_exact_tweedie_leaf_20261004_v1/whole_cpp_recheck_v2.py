"""Whole-audit independent native trace check, no runner/helper import or fit."""
import json,math
import numpy as np
def safeguarded_newton(A,B):
    # Terminate a converged Newton step BEFORE updating its bracket. Otherwise
    # a rounded stationary proposal equals a boundary and spuriously bisects.
    x=0.;lo=-64.;hi=64.
    for _ in range(100):
        g=-A*math.exp(-x/2)+B*math.exp(x/2)+x
        h=.5*(A*math.exp(-x/2)+B*math.exp(x/2))+1
        proposed=x-g/h
        if abs(g/h)<=1e-16 or proposed==x:
            break
        if g>0:hi=x
        else:lo=x
        x=proposed if lo<proposed<hi else (lo+hi)/2
    return x

def independent_cpp(path,booster,tr,bs):
    trees=booster.dump_model()['tree_info'];membership=booster.predict(tr[bs],pred_leaf=True)
    if membership.ndim==1:membership=membership[:,None]
    labels=tr.sub_ec.to_numpy().astype(np.float32).astype(float)
    prior=np.full(len(tr),math.log(math.fsum(map(float,labels))/len(labels)))
    groups=[]
    for line in path.open(encoding='utf-8'):
        item=json.loads(line)
        if item['kind']=='bag':groups.append([item,[]])
        else:assert item['kind']=='leaf' and groups;groups[-1][1].append(item)
    assert len(groups)==len(trees)
    maxF=maxAB=maxRoot=0.
    for t,(bag,leaves) in enumerate(groups):
        assert bag['iteration']==t
        flat=[int(i) for leaf in leaves for i in leaf['ids']]
        assert len(flat)==len(set(flat)) and sorted(flat)==sorted(bag['ids'])
        assert 1<len(leaves)<=31 and len(leaves)==trees[t]['num_leaves']
        nodes=[trees[t]['tree_structure']];values={}
        while nodes:
            node=nodes.pop()
            if 'leaf_index' in node:values[node['leaf_index']]=node['leaf_value']
            else:nodes.extend([node['left_child'],node['right_child']])
        seen=set()
        for leaf in leaves:
            ix=np.asarray(leaf['ids'],int);assert ((ix>=0)&(ix<len(tr))).all()
            yy=np.asarray(leaf['y']);F=np.asarray(leaf['F']);w=np.asarray(leaf['w'])
            assert np.array_equal(yy,labels[ix]) and np.array_equal(w,np.ones(len(ix)))
            maxF=max(maxF,float(np.max(np.abs(F-prior[ix]))));assert maxF<=1e-12
            A=math.fsum(float(y)*math.exp(-float(f)*.5) for y,f in zip(yy,F));B=math.fsum(math.exp(float(f)*.5) for f in F)
            maxAB=max(maxAB,abs(A-leaf['A']),abs(B-leaf['B']))
            assert abs(A-leaf['A'])<=1e-12*max(1,A) and abs(B-leaf['B'])<=1e-12*max(1,B)
            z=float(leaf['root']);neg=A*math.exp(-z/2);pos=B*math.exp(z/2)
            residual=-neg+pos+z;scale=max(1,neg+pos+abs(z))
            assert math.isfinite(z) and abs(residual)<=1e-12*scale
            assert 2*neg+2*pos+.5*z*z<=2*A+2*B+1e-12*max(1,2*A+2*B)
            # Different solver: safeguarded Newton on the strictly increasing derivative.
            x=safeguarded_newton(A,B)
            maxRoot=max(maxRoot,abs(x-z));assert maxRoot<=1e-12
            index=set(map(int,membership[ix,t]));assert len(index)==1;index=index.pop();assert index not in seen;seen.add(index)
            expected=.03*z+(prior[0] if t==0 else 0.)
            assert abs(values[index]-expected)<=1e-12
        assert seen==set(values)
        output=np.asarray([values[int(i)] for i in membership[:,t]])
        prior=output if t==0 else prior+output
    error=float(np.max(np.abs(prior-booster.predict(tr[bs],raw_score=True))));assert error<=1e-12
    return dict(status='PASS',trees=len(trees),max_F_error=maxF,max_AB_error=maxAB,max_root_error=maxRoot,max_final_raw_error=error)
