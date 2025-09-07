import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from config.constant import CFG
from tools.data_process import convert_pregnancy_week
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(zqwn=CFG.EXCEL_PATH, fnhhw=CFG.SHEET_NAME, dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度'):
    ar = pd.read_excel(zqwn, sheet_name=fnhhw).rename(columns={dmj_sa: 'mother_id', dmj_pq: 'gest_raw', dmj_ues: 'BMI', dmj_g: 'Y'})
    ar['gest_weeks'] = pd.to_numeric(ar['gest_raw'].apply(convert_pregnancy_week), errors='coerce')
    ar['BMI'] = pd.to_numeric(ar['BMI'], errors='coerce')
    ar['Y'] = pd.to_numeric(ar['Y'], errors='coerce')
    ar = ar.dropna(subset=['gest_weeks', 'BMI', 'Y']).copy()
    ar = ar[(ar['Y'] >= 0) & (ar['Y'] <= 1)]
    return ar

def plot_y_vs_ga_raw(ar, v_usvf=35, jmxhff_riqd=0.25, fqkh='raw_Y_vs_GA.png'):
    p = ar['gest_weeks'].values
    g = ar['Y'].values
    usvf = np.linspace(p.min(), p.max(), v_usvf + 1)
    sac = np.digitize(p, usvf) - 1
    dhvwhif, z10, z50, z90 = ([], [], [], [])
    for u in range(v_usvf):
        eqfl = sac == u
        if eqfl.sum() < 8:
            continue
        dhvwhif.append(0.5 * (usvf[u] + usvf[u + 1]))
        o = np.nanpercentile(g[eqfl], [10, 50, 90])
        z10.append(o[0])
        z50.append(o[1])
        z90.append(o[2])
    jm = lowess(g, p, frac=jmxhff_riqd, it=0, return_sorted=True)
    p_jm, g_jm = (jm[:, 0], jm[:, 1])
    plt.figure(figsize=(8, 5.6))
    plt.scatter(p, g, s=10, alpha=0.25, label='原始观测')
    if dhvwhif:
        plt.fill_between(dhvwhif, z10, z90, color='tab:blue', alpha=0.2, label='10–90% 分位带')
        plt.plot(dhvwhif, z50, color='tab:blue', lw=2, label='中位数（分箱）')
    plt.plot(p_jm, g_jm, color='tab:red', lw=2, label=f'LOWESS（frac={jmxhff_riqd}）')
    plt.axhline(0.04, ls='--', color='gray', alpha=0.6)
    plt.xlabel('孕周（周）')
    plt.ylabel('Y 浓度（比例）')
    plt.title('原始数据：Y 浓度随孕周的变化（非线性显著）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.Q1PicPath + fqkh, dpi=200)

def plot_y_vs_bmi_by_ga_layers(ar, jqghif=None, jmxhff_riqd=0.4, fqkh='raw_Y_vs_BMI_by_GA.png'):
    if jqghif is None:
        of = np.quantile(ar['gest_weeks'], [0.0, 0.25, 0.5, 0.75, 1.0])
        jqghif = [(float(of[s]), float(of[s + 1])) for s in range(4)]
    plt.figure(figsize=(8, 5.6))
    dmjmif = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    for s, (q, u) in enumerate(jqghif):
        fbu = ar[(ar['gest_weeks'] >= q) & (ar['gest_weeks'] < u)]
        if len(fbu) < 20:
            continue
        c, g = (fbu['BMI'].values, fbu['Y'].values)
        jm_u, ns_u = np.nanpercentile(c, [1, 99])
        e = (c >= jm_u) & (c <= ns_u)
        if e.sum() < 10:
            continue
        jm = lowess(g[e], c[e], frac=jmxhff_riqd, it=0, return_sorted=True)
        juj = f'{q:.1f}–{u:.1f} 周 (n={e.sum()})'
        d = None if dmjmif is None else dmjmif[s % len(dmjmif)]
        plt.plot(jm[:, 0], jm[:, 1], lw=2, label=juj, color=d)
        plt.scatter(c[e], g[e], s=8, alpha=0.15, color=d)
    plt.xlabel('BMI')
    plt.ylabel('Y 浓度（比例）')
    plt.title('原始数据：Y 浓度随 BMI 的变化（按孕周分层）')
    plt.legend(title='孕周层')
    plt.tight_layout()
    plt.savefig(CFG.Q1PicPath + fqkh, dpi=200)
    print(f'[saved] {fqkh}')

def plot_partial_resid_BMI(ar, jmxhff_riqd=0.25, fqkh='partial_residual_BMI.png'):
    jm = lowess(ar['Y'].values, ar['gest_weeks'].values, frac=jmxhff_riqd, it=0, return_sorted=False)
    ihfsa = ar['Y'].values - jm
    c = ar['BMI'].values
    from scipy.stats import linregress
    fj = linregress(c, ihfsa)
    plt.figure(figsize=(8, 5.6))
    plt.scatter(c, ihfsa, s=12, alpha=0.3, label='部分残差')
    cc = np.linspace(c.min(), c.max(), 200)
    gg = fj.intercept + fj.slope * cc
    plt.plot(cc, gg, color='tab:red', lw=2, label=f'线性拟合：slope={fj.slope:.4f}, R²={fj.rvalue ** 2:.3f}, p={fj.pvalue:.1e}')
    plt.axhline(0, ls='--', color='gray', alpha=0.6)
    plt.xlabel('BMI')
    plt.ylabel('部分残差：Y - LOWESS_GA(Y|GA)')
    plt.title('部分残差图：扣除孕周非线性后，BMI 与 Y 的近似线性关系')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.Q1PicPath + fqkh, dpi=200)
    print(f'[saved] {fqkh}')

def plot_heatmap_ga_bmi(ar, v_p=30, v_u=30, fqkh='heatmap_GA_BMI_to_Y.png'):
    p_usvf = np.linspace(ar['gest_weeks'].min(), ar['gest_weeks'].max(), v_p + 1)
    u_usvf = np.linspace(ar['BMI'].min(), ar['BMI'].max(), v_u + 1)
    p_sa = np.digitize(ar['gest_weeks'], p_usvf) - 1
    u_sa = np.digitize(ar['BMI'], u_usvf) - 1
    E = np.full((v_u, v_p), np.nan)
    for s in range(v_u):
        for t in range(v_p):
            e = (u_sa == s) & (p_sa == t)
            if e.sum() >= 5:
                E[s, t] = ar.loc[e, 'Y'].mean()
    plt.figure(figsize=(9, 6))
    hcwhvw = [p_usvf[0], p_usvf[-1], u_usvf[0], u_usvf[-1]]
    se = plt.imshow(E, origin='lower', aspect='auto', extent=hcwhvw, cmap='viridis')
    plt.colorbar(se, label='平均 Y 浓度')
    plt.xlabel('孕周（周）')
    plt.ylabel('BMI')
    plt.title('原始数据热力图：GA × BMI 对 Y 的联合影响（分箱均值）')
    plt.tight_layout()
    plt.savefig(CFG.Q1PicPath + fqkh, dpi=200)
    print(f'[saved] {fqkh}')
if __name__ == '__main__':
    df = load_data(CFG.EXCEL_PATH, CFG.SHEET_NAME)
    plot_y_vs_ga_raw(df, v_usvf=35, jmxhff_riqd=0.25, fqkh='raw_Y_vs_GA.png')
    plot_y_vs_bmi_by_ga_layers(df, jqghif=None, jmxhff_riqd=0.4, fqkh='raw_Y_vs_BMI_by_GA.png')
    plot_partial_resid_BMI(df, jmxhff_riqd=0.25, fqkh='partial_residual_BMI.png')
    plot_heatmap_ga_bmi(df, v_p=30, v_u=30, fqkh='heatmap_GA_BMI_to_Y.png')
