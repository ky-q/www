# -*- coding: utf-8 -*-
"""
原始数据可视化：为采用 GAMM 提供证据
- 图1：Y vs 孕周（散点 + 中位线 + 10-90%分位带 + LOWESS）
- 图2：Y vs BMI（按孕周区间分层的多条 LOWESS 曲线）
- 图3：部分残差图  e = Y - LOWESS_GA(Y|GA)  vs BMI  + 线性拟合
- 图4：GA×BMI -> Y 的热力图（按原始数据分箱均值）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

FILE_PATH = "./data/附件.xlsx"
SHEET = "男胎检测数据"

# ---------- 工具 ----------
def convert_pregnancy_week(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower().replace("＋", "+")
    try:
        if "w+" in s:
            w, d = s.split("w+")
            return float(w) + (float(d) if d!="" else 0.0)/7.0
        if "w" in s:
            return float(s.replace("w",""))
        if "+" in s:
            w, d = s.split("+")
            return float(w) + float(d)/7.0
        return float(s)
    except:
        return np.nan

def load_data(path=FILE_PATH, sheet=SHEET,
              col_id="孕妇代码", col_ga="检测孕周", col_bmi="孕妇BMI", col_y="Y染色体浓度"):
    df = pd.read_excel(path, sheet_name=sheet).rename(
        columns={col_id:"mother_id", col_ga:"gest_raw", col_bmi:"BMI", col_y:"Y"}
    )
    df["gest_weeks"] = pd.to_numeric(df["gest_raw"].apply(convert_pregnancy_week), errors="coerce")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    # 合理范围清理
    df = df.dropna(subset=["gest_weeks","BMI","Y"]).copy()
    df = df[(df["Y"]>=0) & (df["Y"]<=1)]
    return df

# ---------- 图1：Y vs 孕周 ----------
def plot_y_vs_ga_raw(df, n_bins=35, lowess_frac=0.25, save="raw_Y_vs_GA.png"):
    g = df["gest_weeks"].values
    y = df["Y"].values

    # 分箱的中位数 & 分位带
    bins = np.linspace(g.min(), g.max(), n_bins+1)
    idx = np.digitize(g, bins) - 1
    centers, p10, p50, p90 = [], [], [], []
    for b in range(n_bins):
        mask = (idx==b)
        if mask.sum() < 8:  # 太稀疏就跳过
            continue
        centers.append(0.5*(bins[b]+bins[b+1]))
        q = np.nanpercentile(y[mask], [10,50,90])
        p10.append(q[0]); p50.append(q[1]); p90.append(q[2])

    # LOWESS 平滑（原始数据）
    lo = lowess(y, g, frac=lowess_frac, it=0, return_sorted=True)
    g_lo, y_lo = lo[:,0], lo[:,1]

    plt.figure(figsize=(8,5.6))
    plt.scatter(g, y, s=10, alpha=0.25, label="原始观测")
    if centers:
        plt.fill_between(centers, p10, p90, color="tab:blue", alpha=0.2, label="10–90% 分位带")
        plt.plot(centers, p50, color="tab:blue", lw=2, label="中位数（分箱）")
    plt.plot(g_lo, y_lo, color="tab:red", lw=2, label=f"LOWESS（frac={lowess_frac}）")
    plt.axhline(0.04, ls="--", color="gray", alpha=0.6)
    plt.xlabel("孕周（周）"); plt.ylabel("Y 浓度（比例）")
    plt.title("原始数据：Y 浓度随孕周的变化（非线性显著）")
    plt.legend(); plt.tight_layout(); plt.savefig(save, dpi=200)
    print(f"[saved] {save}")

# ---------- 图2：Y vs BMI（孕周分层的 LOWESS） ----------
def plot_y_vs_bmi_by_ga_layers(df, layers=None, lowess_frac=0.4, save="raw_Y_vs_BMI_by_GA.png"):
    """
    layers: 孕周层的区间列表，如 [(11,14),(14,18),(18,22),(22,26)]
            若为 None，自动按四分位生成四层
    """
    if layers is None:
        qs = np.quantile(df["gest_weeks"], [0.00, 0.25, 0.50, 0.75, 1.00])
        layers = [(float(qs[i]), float(qs[i+1])) for i in range(4)]

    plt.figure(figsize=(8,5.6))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    for i,(a,b) in enumerate(layers):
        sub = df[(df["gest_weeks"]>=a) & (df["gest_weeks"]<b)]
        if len(sub) < 20: 
            continue
        x, y = sub["BMI"].values, sub["Y"].values
        # 为避免 BMI 离群点影响 LOWESS，裁剪到 1%-99% 分位
        lo_b, hi_b = np.nanpercentile(x, [1,99])
        m = (x>=lo_b) & (x<=hi_b)
        if m.sum() < 10: 
            continue
        # LOWESS
        lo = lowess(y[m], x[m], frac=lowess_frac, it=0, return_sorted=True)
        lbl = f"{a:.1f}–{b:.1f} 周 (n={m.sum()})"
        c = None if colors is None else colors[i % len(colors)]
        plt.plot(lo[:,0], lo[:,1], lw=2, label=lbl, color=c)
        plt.scatter(x[m], y[m], s=8, alpha=0.15, color=c)

    plt.xlabel("BMI"); plt.ylabel("Y 浓度（比例）")
    plt.title("原始数据：Y 浓度随 BMI 的变化（按孕周分层）")
    plt.legend(title="孕周层")
    plt.tight_layout(); plt.savefig(save, dpi=200)
    print(f"[saved] {save}")

# ---------- 图3：部分残差（把孕周非线性剥离后看 BMI） ----------
def plot_partial_resid_BMI(df, lowess_frac=0.25, save="partial_residual_BMI.png"):
    # 1) 先用 LOWESS 拟合 Y~GA，得到 f(GA)
    lo = lowess(df["Y"].values, df["gest_weeks"].values, frac=lowess_frac, it=0, return_sorted=False)
    resid = df["Y"].values - lo  # e = Y - f(GA)
    x = df["BMI"].values

    # 2) 线性拟合 resid ~ BMI（给出斜率和R^2）
    from scipy.stats import linregress
    sl = linregress(x, resid)

    plt.figure(figsize=(8,5.6))
    plt.scatter(x, resid, s=12, alpha=0.3, label="部分残差")
    xx = np.linspace(x.min(), x.max(), 200)
    yy = sl.intercept + sl.slope * xx
    plt.plot(xx, yy, color="tab:red", lw=2, label=f"线性拟合：slope={sl.slope:.4f}, R²={sl.rvalue**2:.3f}, p={sl.pvalue:.1e}")
    plt.axhline(0, ls="--", color="gray", alpha=0.6)
    plt.xlabel("BMI"); plt.ylabel("部分残差：Y - LOWESS_GA(Y|GA)")
    plt.title("部分残差图：扣除孕周非线性后，BMI 与 Y 的近似线性关系")
    plt.legend(); plt.tight_layout(); plt.savefig(save, dpi=200)
    print(f"[saved] {save}")

# ---------- 图4：GA×BMI -> Y 的热力图（原始数据分箱均值） ----------
def plot_heatmap_ga_bmi(df, n_g=30, n_b=30, save="heatmap_GA_BMI_to_Y.png"):
    g_bins = np.linspace(df["gest_weeks"].min(), df["gest_weeks"].max(), n_g+1)
    b_bins = np.linspace(df["BMI"].min(), df["BMI"].max(), n_b+1)
    g_id = np.digitize(df["gest_weeks"], g_bins)-1
    b_id = np.digitize(df["BMI"], b_bins)-1

    M = np.full((n_b, n_g), np.nan)  # 行对应 BMI 方向
    for i in range(n_b):
        for j in range(n_g):
            m = (b_id==i) & (g_id==j)
            if m.sum() >= 5:  # 至少5个样本才取均值，避免噪声
                M[i, j] = df.loc[m, "Y"].mean()

    # 画热力图
    plt.figure(figsize=(9,6))
    extent = [g_bins[0], g_bins[-1], b_bins[0], b_bins[-1]]
    im = plt.imshow(M, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    plt.colorbar(im, label="平均 Y 浓度")
    plt.xlabel("孕周（周）"); plt.ylabel("BMI")
    plt.title("原始数据热力图：GA × BMI 对 Y 的联合影响（分箱均值）")
    plt.tight_layout(); plt.savefig(save, dpi=200)
    print(f"[saved] {save}")

# ---------- 入口 ----------
if __name__ == "__main__":
    df = load_data(FILE_PATH, SHEET)

    # 图1：Y vs 孕周（能直观看出“非线性”）
    plot_y_vs_ga_raw(df, n_bins=35, lowess_frac=0.25, save="raw_Y_vs_GA.png")

    # 图2：Y vs BMI（分孕周层看，避免被孕周混淆）
    # 也可自定义层：layers=[(11,14),(14,18),(18,22),(22,26)]
    plot_y_vs_bmi_by_ga_layers(df, layers=None, lowess_frac=0.4, save="raw_Y_vs_BMI_by_GA.png")

    # 图3：部分残差（把孕周的非线性剥离后，BMI 显现出近似线性负效应）
    plot_partial_resid_BMI(df, lowess_frac=0.25, save="partial_residual_BMI.png")

    # 图4：GA×BMI -> Y 的热力图（展示联合关系的粗粒度形态）
    plot_heatmap_ga_bmi(df, n_g=30, n_b=30, save="heatmap_GA_BMI_to_Y.png")
