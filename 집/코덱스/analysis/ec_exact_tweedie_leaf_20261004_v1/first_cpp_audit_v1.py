"""Independent Python check of native exact renewal trace; no fitting."""
import json,math
import numpy as np
def audit_trace(path,model,tr,features):
    trees=model.booster_.dump_model()['tree_info'];groups=[]
    for line in path.open(encoding='utf-8'):
        r=json.loads(line)
        if r['kind']=='bag':groups.append([r,[]])
        else:assert r['kind']=='leaf' and groups;groups[-1][1].append(r)
    assert len(groups)==len(trees)
    membership=model.booster_.predict(tr[features],pred_leaf=True)
    if membership.ndim==1:membership=membership[:,None]
    target=tr.sub_ec.to_numpy().astype(np.float32).astype(float)
    F=np.full(len(tr),math.log(math.fsum(target)/len(target)))
    max_F=max_sum=max_root=0.;summary=[]
    for ti,(bag,leaves) in enumerate(groups):
        assert bag['iteration']==ti
        flat=[i for leaf in leaves for i in leaf['ids']]
        assert len(flat)==len(set(flat)) and sorted(flat)==sorted(bag['ids'])
        assert len(leaves)==trees[ti]['num_leaves'] and 1<=len(leaves)<=31
        def collect(node):
            if 'leaf_index' in node:return {node['leaf_index']:node['leaf_value']}
            return collect(node['left_child'])|collect(node['right_child'])
        values=collect(trees[ti]['tree_structure']);seen=set()
        for leaf in leaves:
            ix=np.asarray(leaf['ids'],int);score=np.asarray(leaf['F']);y=np.asarray(leaf['y']);w=np.asarray(leaf['w'])
            assert np.array_equal(y,target[ix]) and np.array_equal(w,np.ones(len(ix)))
            max_F=max(max_F,float(np.max(np.abs(score-F[ix]))));assert max_F<=1e-12
            A=math.fsum(float(a*math.exp(-b/2)) for a,b in zip(y,score));B=math.fsum(math.exp(float(b)/2) for b in score)
            max_sum=max(max_sum,abs(A-leaf['A']),abs(B-leaf['B']))
            assert abs(A-leaf['A'])<=1e-12*max(1,A) and abs(B-leaf['B'])<=1e-12*max(1,B)
            lo,hi=-64.,64.
            for _ in range(120):
                mid=(lo+hi)/2
                if -A*math.exp(-mid/2)+B*math.exp(mid/2)+mid>0:hi=mid
                else:lo=mid
            root=(lo+hi)/2;max_root=max(max_root,abs(root-leaf['root']));assert max_root<=1e-12
            li=np.unique(membership[ix,ti]);assert len(li)==1;li=int(li[0]);assert li not in seen;seen.add(li)
            expected=.03*root+(F[0] if ti==0 else 0.)
            assert abs(values[li]-expected)<=1e-12
        assert seen==set(values)
        F=(np.asarray([values[int(li)] for li in membership[:,ti]]) if ti==0 else F+np.asarray([values[int(li)] for li in membership[:,ti]]))
        summary.append(dict(iteration=ti,inbag_count=len(flat),leaf_count=len(leaves)))
    end=model.booster_.predict(tr[features],raw_score=True)
    assert np.max(np.abs(end-F))<=1e-12
    model.booster_._Booster__is_predicted_cur_iter[0]=False
    cache=model.booster_._Booster__inner_predict(data_idx=0)
    assert np.max(np.abs(cache-np.exp(F)))<=1e-12
    return dict(status='PASS',tree_count=len(trees),groups=summary,max_F_error=max_F,max_sum_error=max_sum,max_root_error=max_root,cache_error=float(np.max(np.abs(cache-np.exp(F)))))
