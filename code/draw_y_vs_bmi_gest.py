import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from config.constant import CFG
from tools.data_process import convert_pregnancy_week
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
def load_data(path=CFG.EXCEL_PATH, sheet=CFG.SHEET_NAME,
              col_id="孕妇代码", col_ga="检测孕周", col_bmi="孕妇BMI", col_y="Y染色体浓度"):
    df = pd.read_excel(path, sheet_name=sheet).rename(
        columns={col_id:"mother_id", col_ga:"gest_raw", col_bmi:"BMI", col_y:"Y"}
    )
    df["gest_weeks"] = pd.to_numeric(df["gest_raw"].apply(convert_pregnancy_week), errors="coerce")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["gest_weeks","BMI","Y"]).copy()
    df = df[(df["Y"]>=0) & (df["Y"]<=1)]
    return df
def plot_y_vs_ga_raw(df, n_bins=35, lowess_frac=0.25, save="raw_Y_vs_GA.png"):
    g = df["gest_weeks"].values
    y = df["Y"].values
    bins = np.linspace(g.min(), g.max(), n_bins+1)
    idx = np.digitize(g, bins) - 1
    centers, p10, p50, p90 = [], [], [], []
    for b in range(n_bins):
        mask = (idx==b)
        if mask.sum() < 8:
            continue
        centers.append(0.5*(bins[b]+bins[b+1]))
        q = np.nanpercentile(y[mask], [10,50,90])
        p10.append(q[0]); p50.append(q[1]); p90.append(q[2])
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
    plt.legend(); plt.tight_layout(); plt.savefig(CFG.Q1PicPath + save, dpi=200)
def plot_y_vs_bmi_by_ga_layers(df, layers=None, lowess_frac=0.4, save="raw_Y_vs_BMI_by_GA.png"):
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
        lo_b, hi_b = np.nanpercentile(x, [1,99])
        m = (x>=lo_b) & (x<=hi_b)
        if m.sum() < 10:
            continue
        lo = lowess(y[m], x[m], frac=lowess_frac, it=0, return_sorted=True)
        lbl = f"{a:.1f}–{b:.1f} 周 (n={m.sum()})"
        c = None if colors is None else colors[i % len(colors)]
        plt.plot(lo[:,0], lo[:,1], lw=2, label=lbl, color=c)
        plt.scatter(x[m], y[m], s=8, alpha=0.15, color=c)
    plt.xlabel("BMI"); plt.ylabel("Y 浓度（比例）")
    plt.title("原始数据：Y 浓度随 BMI 的变化（按孕周分层）")
    plt.legend(title="孕周层")
    plt.tight_layout(); plt.savefig(CFG.Q1PicPath + save, dpi=200)
    print(f"[saved] {save}")
def plot_partial_resid_BMI(df, lowess_frac=0.25, save="partial_residual_BMI.png"):
    lo = lowess(df["Y"].values, df["gest_weeks"].values, frac=lowess_frac, it=0, return_sorted=False)
    resid = df["Y"].values - lo
    x = df["BMI"].values
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
    plt.legend(); plt.tight_layout(); plt.savefig(CFG.Q1PicPath + save, dpi=200)
    print(f"[saved] {save}")
def plot_heatmap_ga_bmi(df, n_g=30, n_b=30, save="heatmap_GA_BMI_to_Y.png"):
    g_bins = np.linspace(df["gest_weeks"].min(), df["gest_weeks"].max(), n_g+1)
    b_bins = np.linspace(df["BMI"].min(), df["BMI"].max(), n_b+1)
    g_id = np.digitize(df["gest_weeks"], g_bins)-1
    b_id = np.digitize(df["BMI"], b_bins)-1
    M = np.full((n_b, n_g), np.nan)
    for i in range(n_b):
        for j in range(n_g):
            m = (b_id==i) & (g_id==j)
            if m.sum() >= 5:
                M[i, j] = df.loc[m, "Y"].mean()
    plt.figure(figsize=(9,6))
    extent = [g_bins[0], g_bins[-1], b_bins[0], b_bins[-1]]
    im = plt.imshow(M, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    plt.colorbar(im, label="平均 Y 浓度")
    plt.xlabel("孕周（周）"); plt.ylabel("BMI")
    plt.title("原始数据热力图：GA × BMI 对 Y 的联合影响（分箱均值）")
    plt.tight_layout(); plt.savefig(CFG.Q1PicPath + save, dpi=200)
    print(f"[saved] {save}")
if __name__ == "__main__":
    df = load_data(CFG.EXCEL_PATH, CFG.SHEET_NAME)
    plot_y_vs_ga_raw(df, n_bins=35, lowess_frac=0.25, save="raw_Y_vs_GA.png")
    plot_y_vs_bmi_by_ga_layers(df, layers=None, lowess_frac=0.4, save="raw_Y_vs_BMI_by_GA.png")
    plot_partial_resid_BMI(df, lowess_frac=0.25, save="partial_residual_BMI.png")
    plot_heatmap_ga_bmi(df, n_g=30, n_b=30, save="heatmap_GA_BMI_to_Y.png")
