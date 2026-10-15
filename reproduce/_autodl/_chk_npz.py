import numpy as np
for ds in ["arab", "rice", "uav"]:
    z = np.load("/root/autodl-tmp/qkn/outputs/reproduce/knode_dynamic_baselines_full_%s_v2_preds.npz" % ds)
    ks = sorted(z.keys())
    print(ds, "keys:", len(ks), "| sample:", ks[:3], "| iter9 keys:", [k for k in ks if "iter9" in k][:3])
    print("   shapes:", ks[0], z[ks[0]].shape)