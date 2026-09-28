import env, numpy as np
z=np.load('local/oof_temp_diag.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype, z[k][:3])
