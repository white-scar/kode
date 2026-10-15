import sys, importlib
sys.path.insert(0, '/root/autodl-tmp/qkn/reproduce')
import knode_hyperparam_ablation as H
ds = sys.argv[1]
grid = sys.argv[2].split()
H.aggregate(ds, grid, [f'_hp_{c}' for c in grid])