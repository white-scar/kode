# -*- coding: utf-8 -*-
"""stats_paired_bootstrap_fdr.py — 以 fold-set 为独立重复单元的配对统计 (Kode vs 作者方法)

实验单位: fold-set (n = 10)。性状与时间点视为重复测量, 不作独立样本。
  1) 总体差异: 先在每个 fold-set 内对 (trait x DAS) 聚合得到该折精度 -> 10 个配对差
     -> 均值差 / 95% CI / 双侧 p (percentile bootstrap, 10000 次重采样 fold-set)
  2) 逐性状: 每个性状内先逐折对 DAS 平均 -> 10 个配对差 -> 同一 bootstrap
     -> 再对全部性状的 p 做 Benjamini-Hochberg FDR, 报 q < 0.05 的显著性状数

输入: pic-code/pic2 的 unified_norm_author_vs_kode_per_foldset.csv (Series x Fold x DAS x trait)
      (当前口径: 训练折 min-max; Kode = affine, 无 skip)
输出 (outputs/stats/):
  paired_bootstrap_summary.csv                总体配对 bootstrap 汇总 (PCC)
  paired_bootstrap_summary_mse.csv            同上, 指标换成 MSE (mean_diff = Kode - 作者, 负值=Kode 更好)
  {ds}_per_trait_boot_fdr.csv                 逐性状 mean_diff/CI/p/win/q_fdr
  {ds}_per_trait_win_rate_Kode_vs_{cmp}.csv   逐性状胜率
用法: python stats_paired_bootstrap_fdr.py [--n_boot 10000] [--seed 42]
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(ROOT, ".."))
IN = {
    "maize":       os.path.join(ROOT, "reproduce", "pic-code", "pic2", "unified_norm_author_vs_kode_per_foldset.csv"),
    "arabidopsis": os.path.join(ROOT, "reproduce", "pic-code", "pic2", "arab-E1", "unified_norm_author_vs_kode_per_foldset.csv"),
    "rice":        os.path.join(ROOT, "reproduce", "pic-code", "pic2", "rice-E2", "unified_norm_author_vs_kode_per_foldset.csv"),
}
OUT = os.path.join(ROOT, "outputs", "stats")
# (比较名, Kode Series, 作者 Series, 简称)
CMPS = [("Kode_ode_hyper_vs_tp1_rec",  "Kode_rec",  "author_tp1_rec",  "tp1_rec"),
        ("Kode_iter_hyper_vs_tp1_iter", "Kode_iter", "author_tp1_iter", "tp1_iter")]


def boot(d, n_boot, rng):
    """d: 1-D 配对差 (长度 = fold 数). 返回 mean, lo, hi, 双侧 p."""
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2.0 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p); o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    summ, summ_mse = [], []
    for ds, path in IN.items():
        L = pd.read_csv(path)
        for cmpn, kode, dgp, short in CMPS:
            k = L[L.Series == kode]; g = L[L.Series == dgp]
            if k.empty or g.empty:
                continue
            # ---- 总体: 逐折对 (trait x DAS) 平均 ----
            kf = k.groupby("Fold")["PCC"].mean(); gf = g.groupby("Fold")["PCC"].mean()
            folds = sorted(set(kf.index) & set(gf.index))
            d = (kf.loc[folds].values - gf.loc[folds].values)
            md, lo, hi, p = boot(d, a.n_boot, rng)
            summ.append(dict(mean_diff=md, lo=lo, hi=hi, p=p, n_folds=len(folds),
                             kode_mean=float(kf.loc[folds].mean()), dgp_mean=float(gf.loc[folds].mean()),
                             dataset=ds, comparison=cmpn))
            km = k.groupby("Fold")["MSE"].mean(); gm = g.groupby("Fold")["MSE"].mean()
            dm = (km.loc[folds].values - gm.loc[folds].values)
            md2, lo2, hi2, p2 = boot(dm, a.n_boot, rng)
            summ_mse.append(dict(mean_diff=md2, lo=lo2, hi=hi2, p=p2, n_folds=len(folds),
                                 kode_mean=float(km.loc[folds].mean()), dgp_mean=float(gm.loc[folds].mean()),
                                 dataset=ds, comparison=cmpn))
            # ---- 逐性状: 每折先对 DAS 平均 ----
            kt = k.groupby(["trait", "Fold"])["PCC"].mean().unstack()
            gt = g.groupby(["trait", "Fold"])["PCC"].mean().unstack()
            common = [t for t in kt.index if t in set(gt.index)]
            rows = []
            for t in common:
                dd = (kt.loc[t, folds].values - gt.loc[t, folds].values)
                m_, l_, h_, p_ = boot(dd, a.n_boot, rng)
                rows.append(dict(trait=t, mean_diff=m_, lo=l_, hi=h_, p=p_,
                                 kode_mean=float(kt.loc[t, folds].mean()),
                                 dgp_mean=float(gt.loc[t, folds].mean()),
                                 win=bool(m_ > 0)))
            tr = pd.DataFrame(rows)
            tr["q_fdr"] = bh(tr["p"].values)
            tr["dataset"] = ds
            tr[["trait", "mean_diff", "lo", "hi", "p", "kode_mean", "dgp_mean", "win", "q_fdr", "dataset"]] \
                .to_csv(os.path.join(OUT, f"{ds}_per_trait_boot_fdr.csv"), index=False)
            tr[["trait", "kode_mean", "dgp_mean"]].assign(
                Kode_win_rate=[float((kt.loc[t, folds].values > gt.loc[t, folds].values).mean())
                               for t in tr["trait"]], dataset=ds) \
                .to_csv(os.path.join(OUT, f"{ds}_per_trait_win_rate_Kode_vs_{short}.csv"), index=False)
            print("[%s %s] n_folds=%d  Kode=%.4f dgp=%.4f  diff=%+.4f 95%%CI[%.4f, %.4f] p=%.4g | "
                  "win=%d/%d  FDR(q<.05)=%d/%d"
                  % (ds, cmpn, len(folds), kf.loc[folds].mean(), gf.loc[folds].mean(), md, lo, hi, p,
                     int(tr.win.sum()), len(tr), int((tr.q_fdr < 0.05).sum()), len(tr)))

    pd.DataFrame(summ)[["mean_diff", "lo", "hi", "p", "n_folds", "kode_mean", "dgp_mean", "dataset", "comparison"]] \
        .to_csv(os.path.join(OUT, "paired_bootstrap_summary.csv"), index=False)
    pd.DataFrame(summ_mse)[["mean_diff", "lo", "hi", "p", "n_folds", "kode_mean", "dgp_mean", "dataset", "comparison"]] \
        .to_csv(os.path.join(OUT, "paired_bootstrap_summary_mse.csv"), index=False)
    print("\nsaved -> outputs/stats/paired_bootstrap_summary.csv (+ _mse, + per-trait/fdr/win_rate)")


if __name__ == "__main__":
    main()
