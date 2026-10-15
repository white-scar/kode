import pandas as pd
from pathlib import Path
EN = Path("/root/autodl-tmp/qkn/outputs/reproduce/author_all2_rice_innerens/components")
for f in ["iter01_val_phi_enspred_tp1.csv", "iter01_val_R_enspred_tp1.csv"]:
    d = pd.read_csv(EN / f, index_col=0)
    print("#####", f, d.shape)
    print("  columns[:6]:", list(d.columns[:6]))
    print("  index[:3]:", list(d.index[:3]))
print()
import re
t = Path("/root/autodl-tmp/qkn/reproduce/R/tensor_unroller.R").read_text(encoding="utf-8")
print(t[:1200])