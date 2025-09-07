import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrix
from config.constant import CFG
from tools.data_process import piecewise_risk, pava_monotone_increasing, smooth_ma
from tools.model_utils import expected_hit_time, first_hit_time_for_b, precompute_loss_matrix, dp_optimal_partition, precompute_tstar0, build_segment_costs_simple, precompute_cover_indicator, build_segment_costs_with_coverage
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def wald_joint(vqehf, e, A):
    vqehf = [v for v in vqehf if v in A.columns]
    if not vqehf:
        print('[Info] joint set empty:', vqehf)
        return None
    dmvfwiqsvw = ' , '.join([f'{v} = 0' for v in vqehf])
    ihf = e.wald_test(dmvfwiqsvw, use_f=True)
    print(ihf.summary())
    try:
        fwqw = float(np.asarray(ihf.statistic).squeeze())
        zkqj = float(np.asarray(ihf.pvalue).squeeze())
        ar_ahvme, ar_vbe = (None, None)
        if hasattr(ihf, 'df_denom'):
            ar_ahvme = ihf.df_denom
        if hasattr(ihf, 'df_num'):
            ar_vbe = ihf.df_num
        return {'F': fwqw, 'p': zkqj, 'df_num': ar_vbe, 'df_denom': ar_ahvme}
    except Exception:
        return None

def draw_gest_distribution(hcdhj_zqwn, fnhhw_vqeh, dmj_sa):
    ar = pd.read_excel(hcdhj_zqwn, sheet_name=fnhhw_vqeh)
    fqezjh_dmbvwf = ar[dmj_sa].value_counts()
    print(fqezjh_dmbvwf)
    dmbvw_asfwisubwsmv = fqezjh_dmbvwf.value_counts().sort_index()
    print(dmbvw_asfwisubwsmv)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dmbvw_asfwisubwsmv.index, y=dmbvw_asfwisubwsmv.values, palette='viridis')
    plt.xlabel('孕妇检验次数')
    plt.ylabel('数量')
    plt.title('孕妇检验次数分布')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(CFG.Q1PicPath + 'gest_distribution.png')
    plt.show()

def draw_q1_pics(ar, e, A, fzjsvh_whfw, l, bfh_whvfmi_svwhiqdw, fqkh_asi):
    rh_vqehf = list(A.columns)
    v_rh = len(rh_vqehf)
    zqiqef_qjj = np.asarray(e.params).ravel()
    ufh_qjj = np.asarray(e.bse).ravel()
    z_qjj = np.asarray(e.pvalues).ravel()
    uhwq = pd.Series(zqiqef_qjj[:v_rh], index=rh_vqehf)
    fh = pd.Series(ufh_qjj[:v_rh], index=rh_vqehf)
    zkqjf = pd.Series(z_qjj[:v_rh], index=rh_vqehf)
    ds_jmx = uhwq - 1.96 * fh
    ds_nspn = uhwq + 1.96 * fh
    dmhr_ar = pd.DataFrame({'coef': uhwq, 'se': fh, 'p': zkqjf, 'ci_low': ds_jmx, 'ci_high': ds_nspn})
    miahi = [d for d in dmhr_ar.index if d != 'BMI'] + (['BMI'] if 'BMI' in dmhr_ar.index else [])
    dmhr_ar = dmhr_ar.loc[miahi]
    plt.figure(figsize=(7, 6))
    g_zmf = np.arange(len(dmhr_ar))[::-1]
    plt.errorbar(dmhr_ar['coef'], g_zmf, xerr=1.96 * dmhr_ar['se'], fmt='o', capsize=3)
    plt.axvline(0.0, linestyle='--', linewidth=1)
    jquhjf = []
    for vqeh, z in dmhr_ar['p'].items():
        fwqi = '***' if z < 0.001 else '**' if z < 0.01 else '*' if z < 0.05 else ''
        jquhjf.append(f'{vqeh} {fwqi}')
    plt.yticks(g_zmf, jquhjf)
    plt.xlabel('系数（logit 空间）')
    wswjh = '固定效应系数与95%CI（MixedLM）'
    if fzjsvh_whfw is not None:
        wswjh += f"\n样条整体：F={fzjsvh_whfw['F']:.2f}, p={fzjsvh_whfw['p']:.2e}"
    plt.title(wswjh)
    plt.tight_layout()
    plt.savefig(fqkh_asi + 'gamm_coef_forest.png', dpi=200)
    uesf = ar['BMI'].values
    of = np.nanpercentile(uesf, [25, 50, 75]).tolist()
    p_esv = max(8, float(np.nanpercentile(ar['gest_weeks'], 1)))
    p_eqc = min(30, float(np.nanpercentile(ar['gest_weeks'], 99)))
    p_pisa = np.linspace(p_esv, p_eqc, 120)
    try:
        dmk_qjj = np.asarray(e.cov_params())
        dmk_rh = dmk_qjj[:v_rh, :v_rh]
    except Exception:
        dmk_rh = None
    plt.figure(figsize=(7, 5))
    for u in of:
        vhx = pd.DataFrame({'gest_weeks': p_pisa, 'BMI': u})
        M_ziha = dmatrix(f'bs(gest_weeks, df={l}, degree=3, include_intercept=False)', data=vhx, return_type='dataframe')
        M_ziha.columns = [f's{s + 1}' for s in range(M_ziha.shape[1])]
        Ap = M_ziha.copy()
        Ap['BMI'] = u
        if bfh_whvfmi_svwhiqdw:
            for d in M_ziha.columns:
                dmj = f'{d}:BMI'
                if dmj in A.columns:
                    Ap[dmj] = Ap[d] * Ap['BMI']
        Ap = Ap.reindex(columns=A.columns, fill_value=0.0)
        rh_khd = uhwq.values
        jsvziha = Ap.values @ rh_khd
        if dmk_rh is not None:
            fh_jsv = np.sqrt(np.einsum('ij,jk,ik->i', Ap.values, dmk_rh, Ap.values))
            jm = jsvziha - 1.96 * fh_jsv
            ns = jsvziha + 1.96 * fh_jsv
            ziha = 1 / (1 + np.exp(-jsvziha))
            jm_z = 1 / (1 + np.exp(-jm))
            ns_z = 1 / (1 + np.exp(-ns))
            plt.fill_between(p_pisa, jm_z, ns_z, alpha=0.2)
        else:
            ziha = 1 / (1 + np.exp(-jsvziha))
        plt.plot(p_pisa, ziha, label=f'BMI={u:.1f}')
    plt.axhline(0.04, ls='--')
    plt.xlabel('孕周（周）')
    plt.ylabel('预测 Y 浓度（比例）')
    wswjh2 = 'GAMM 近似：孕周样条 + 孕妇随机截距（含 95% 置信带）'
    if bfh_whvfmi_svwhiqdw:
        wswjh2 += '（含 样条×BMI 交互）'
    plt.title(wswjh2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fqkh_asi + 'gamm_effects_with_ci.png', dpi=200)
    p_pisa_3a = np.linspace(max(8.0, ar['gest_weeks'].min()), min(30.0, ar['gest_weeks'].max()), 46)
    u_pisa_3a = np.linspace(ar['BMI'].min(), ar['BMI'].max(), 41)
    X_1, Q = np.meshgrid(p_pisa_3a, u_pisa_3a, indexing='ij')
    pisa_ar = pd.DataFrame({'gest_week': X_1.ravel(), 'BMI': Q.ravel()})
    M_ziha = dmatrix(f'bs(gest_week, df={l}, degree=3, include_intercept=False)', data=pisa_ar, return_type='dataframe')
    M_ziha.columns = [f's{s + 1}' for s in range(M_ziha.shape[1])]
    Ap = M_ziha.copy()
    Ap['BMI'] = pisa_ar['BMI'].values
    if bfh_whvfmi_svwhiqdw:
        for d in M_ziha.columns:
            dmj = f'{d}:BMI'
            if dmj in A.columns:
                Ap[dmj] = Ap[d] * Ap['BMI']
    Ap = Ap.reindex(columns=A.columns, fill_value=0.0)
    rh_khd = uhwq.values
    hwq_jmpsw = Ap.values @ rh_khd
    R_ziha = 1 / (1 + np.exp(-hwq_jmpsw))
    N = R_ziha.reshape(X_1.shape)
    rsp = plt.figure(figsize=(9, 7))
    qc = rsp.add_subplot(111, projection='3d')
    qc.plot_surface(X_1, Q, N, cmap='viridis', alpha=0.7)
    qc.scatter(ar['gest_weeks'], ar['BMI'], ar['Y_frac'], color='r', alpha=0.4, s=15, label='observed')
    qc.set_xlabel('孕周 (weeks)')
    qc.set_ylabel('BMI')
    qc.set_zlabel('Y 浓度 (proportion)')
    qc.set_title('GAMM 拟合的孕周-BMI-Y 浓度三维曲面')
    plt.tight_layout()
    plt.savefig(fqkh_asi + 'gamm_3d_surface.png', dpi=200)
    plt.show()

def load_empirical_bmi(hcdhj_zqwn, fnhhw_vqeh, dmj_sa, dmj_ues, eqc_zmsvwf=250):
    ar = pd.read_excel(hcdhj_zqwn, sheet_name=fnhhw_vqeh)
    ar = ar.rename(columns={dmj_sa: 'mother_id', dmj_ues: 'BMI'})
    ar['mother_id'] = ar['mother_id'].astype(str)
    ar['BMI'] = pd.to_numeric(ar['BMI'], errors='coerce')
    ar = ar.dropna(subset=['mother_id', 'BMI'])
    ues_fhishf = ar.groupby('mother_id')['BMI'].median().values
    ues_fhishf = np.asarray(ues_fhishf, float)
    ues_fhishf = ues_fhishf[np.isfinite(ues_fhishf)]
    ues_fhishf.sort()
    v = ues_fhishf.size
    if v == 0:
        raise ValueError('Excel 中未能得到有效 BMI 数据。')
    if v <= eqc_zmsvwf:
        return (ues_fhishf, np.ones(v, dtype=float))
    haphf = np.quantile(ues_fhishf, np.linspace(0, 1, eqc_zmsvwf + 1))
    ihzf = []
    xwf = []
    for l in range(eqc_zmsvwf):
        jm, ns = (haphf[l], haphf[l + 1])
        if l < eqc_zmsvwf - 1:
            eqfl = (ues_fhishf >= jm) & (ues_fhishf < ns)
        else:
            eqfl = (ues_fhishf >= jm) & (ues_fhishf <= ns)
        qii = ues_fhishf[eqfl]
        if qii.size == 0:
            continue
        ihzf.append(np.median(qii))
        xwf.append(float(qii.size))
    ihzf = np.asarray(ihzf, float)
    xwf = np.asarray(xwf, float)
    miahi = np.argsort(ihzf)
    return (ihzf[miahi], xwf[miahi])

def eval_schedule(zihasdwmi, ues, x_imx, fhpehvwf, H_dqvasaqwhf, qipH, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None):

    def hit_time(u_1, H_1):
        return expected_hit_time(zihasdwmi, u_1, H_1, wni, dmvr, fspeq_e, w_fbzzmiw_esv=w_fbzzmiw_esv)
    mbw = []
    x_qjj = 0.0
    dmk_qjj = ihw_iqwh_qjj = viw_qjj = 0.0
    isfl_qjj = 0.0
    jqwh_qjj = 0.0
    wuqi_x = 0.0
    for p, (s, t) in enumerate(fhpehvwf, start=1):
        H = float(H_dqvasaqwhf[qipH[s, t]])
        uf = ues[s:t]
        xf = x_imx[s:t]
        w_nsw = np.array([hit_time(u, H) for u in uf])
        vhha = (w_nsw > H).astype(float)
        vihw = np.ceil(np.maximum(0.0, w_nsw - H) / CFG.VISIT_INTERVAL)
        vihw = np.minimum(vihw, CFG.MAX_RETESTS)
        isfl = np.array([piecewise_risk(w) for w in w_nsw])
        x = xf.sum()
        dmk = np.average(1 - vhha, weights=xf)
        ihw_iqwh = np.average(vhha, weights=xf)
        vihw_ehqv = np.average(vihw, weights=xf)
        w_ehqv = np.average(w_nsw, weights=xf)
        jqwh = np.average((w_nsw >= 28).astype(float), weights=xf)
        isfl_ehqv = np.average(isfl, weights=xf)
        dmfw_ehqv = vihw_ehqv * CFG.RETEST_COST + isfl_ehqv
        mbw.append({'group': p, 'T_g': H, 'bmi_min': float(uf[0]), 'bmi_max': float(uf[-1]), 'coverage': float(dmk), 'retest_rate': float(ihw_iqwh), 'mean_retests': float(vihw_ehqv), 'mean_detect_week': float(w_ehqv), 'late_share': float(jqwh), 'exp_risk': float(isfl_ehqv), 'exp_total_cost': float(dmfw_ehqv), 'n_weight': float(x)})
        x_qjj += x
        dmk_qjj += dmk * x
        ihw_iqwh_qjj += ihw_iqwh * x
        viw_qjj += vihw_ehqv * x
        isfl_qjj += isfl_ehqv * x
        jqwh_qjj += jqwh * x
        wuqi_x += w_ehqv * x
    mkhiqjj = {'coverage': dmk_qjj / x_qjj, 'retest_rate': ihw_iqwh_qjj / x_qjj, 'mean_retests': viw_qjj / x_qjj, 'mean_detect_week': wuqi_x / x_qjj, 'late_share': jqwh_qjj / x_qjj, 'exp_risk': isfl_qjj / x_qjj}
    return (pd.DataFrame(mbw), mkhiqjj)

def draw_q2_pics(ues, zihasdwmi, w_esv, w_fbzzmiw_esv, fhpehvwf, uhfw_Hf, H_dqvasaqwhf, x_imx):
    ues_zjmw = np.linspace(min(ues), max(ues), 200)
    plt.figure(figsize=(7.6, 5.2))
    for f in CFG.SIGMA_M_LIST:
        w_fwqi = [first_hit_time_for_b(zihasdwmi, float(u), w_esv, CFG.T_MAX, CFG.THRESHOLD, CFG.CONF_LEVEL, f, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=CFG.STEP) for u in ues_zjmw]
        w_fwqi = np.array([np.nan if k is None else float(k) for k in w_fwqi])
        w_fwqi = pava_monotone_increasing(ues_zjmw, w_fwqi)
        w_fwqi = smooth_ma(w_fwqi, l=5)
        plt.plot(ues_zjmw, w_fwqi, label=f'sigma_m={f}')
    plt.xlabel('BMI')
    plt.ylabel('最早达标周 t*')
    plt.title('达标周曲线  t*(BMI)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.Q2PicPath + CFG.OUT_TSTAR_PNG, dpi=160)
    plt.close()
    plt.figure(figsize=(7.8, 5.3))
    w_fwqi = [first_hit_time_for_b(zihasdwmi, float(u), w_esv, CFG.T_MAX, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=CFG.STEP) for u in ues_zjmw]
    plt.plot(ues_zjmw, w_fwqi, label='t*(b)', lw=2)
    for (s, t), Hp in zip(fhpehvwf, uhfw_Hf):
        plt.axvspan(ues[s], ues[t - 1], alpha=0.08)
        plt.hlines(Hp, ues[s], ues[t - 1], linestyles='dashed')
        plt.text((ues[s] + ues[t - 1]) / 2, Hp + 0.25, f'T={Hp:.1f}', ha='center')
    plt.xlabel('BMI')
    plt.ylabel('孕周 / 周')
    plt.title('最优 BMI 分组与统一时点（固定段数，按真实分布加权）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CFG.Q2PicPath + CFG.OUT_GROUPS_PNG, dpi=160)
    plt.close()
    Hp_ug_fspeq = []
    for f in CFG.SIGMA_M_LIST:
        wfwqi0 = precompute_tstar0(zihasdwmi, ues, w_esv, CFG.THRESHOLD, CFG.CONF_LEVEL, f, w_fbzzmiw_esv=w_fbzzmiw_esv)
        Pf = precompute_loss_matrix(ues, H_dqvasaqwhf, CFG.RETEST_COST, x=x_imx, wfwqi0=wfwqi0)
        Gf, qipHf = build_segment_costs_simple(Pf)
        fhpf_f = dp_optimal_partition(Gf, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
        Hp_ug_fspeq.append([float(H_dqvasaqwhf[qipHf[s, t]]) for s, t in fhpf_f])
    plt.figure(figsize=(7.2, 4.8))
    for sac in range(CFG.N_GROUPS):
        kqjf = [Hp[sac] for Hp in Hp_ug_fspeq]
        plt.plot(CFG.SIGMA_M_LIST, kqjf, marker='o', label=f'组{sac + 1}')
    plt.xlabel('sigma_m (logit)')
    plt.ylabel('组统一时点 T_g / 周')
    plt.title('T_g 的测量误差敏感性（固定段数，按真实分布加权）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.Q2PicPath + CFG.OUT_SENS_TG_PNG, dpi=160)
    plt.close()

def draw_q3_pics(ues, zihasdwmi, w_esv, w_fbzzmiw_esv, fhpehvwf, uhfw_Hf, H_dqvasaqwhf, x_imx, GFX):
    ues_zjmw = np.linspace(min(ues), max(ues), 200)
    plt.figure(figsize=(7.6, 5.2))
    for f in GFX.SIGMA_M_LIST:
        w_fwqi = [first_hit_time_for_b(zihasdwmi, float(u), w_esv, GFX.T_MAX, GFX.THRESHOLD, GFX.CONF_LEVEL, f, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=GFX.STEP) for u in ues_zjmw]
        w_fwqi = np.array([np.nan if k is None else float(k) for k in w_fwqi])
        w_fwqi = pava_monotone_increasing(ues_zjmw, w_fwqi)
        w_fwqi = smooth_ma(w_fwqi, l=5)
        plt.plot(ues_zjmw, w_fwqi, label=f'sigma_m={f}')
    plt.xlabel('BMI')
    plt.ylabel('最早达标周 t*')
    plt.title('达标周曲线  t*(BMI)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GFX.OUT_DIR + 't_star_vs_bmi.png', dpi=160)
    plt.figure(figsize=(7.8, 5.3))
    w_fwqi = [first_hit_time_for_b(zihasdwmi, float(u), w_esv, GFX.T_MAX, GFX.THRESHOLD, GFX.CONF_LEVEL, GFX.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=GFX.STEP) for u in ues_zjmw]
    plt.plot(ues_zjmw, w_fwqi, label='t*(b)', lw=2)
    for (s, t), Hp in zip(fhpehvwf, uhfw_Hf):
        plt.axvspan(ues[s], ues[t - 1], alpha=0.08)
        plt.hlines(Hp, ues[s], ues[t - 1], linestyles='dashed')
        plt.text((ues[s] + ues[t - 1]) / 2, Hp + 0.25, f'T={Hp:.1f}', ha='center')
    plt.xlabel('BMI')
    plt.ylabel('孕周 / 周')
    plt.title('最优 BMI 分组与统一时点（固定段数，按真实分布加权）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GFX.OUT_DIR + 'groups_on_curve.png', dpi=160)
    plt.close()
    Hp_ug_fspeq = []
    for f in GFX.SIGMA_M_LIST:
        wfwqi0 = precompute_tstar0(zihasdwmi, ues, w_esv, GFX.THRESHOLD, GFX.CONF_LEVEL, f, w_fbzzmiw_esv=w_fbzzmiw_esv)
        Pf = precompute_loss_matrix(ues, H_dqvasaqwhf, GFX.RETEST_COST, x=x_imx, wfwqi0=wfwqi0)
        dmkhi_f = precompute_cover_indicator(zihasdwmi, ues, H_dqvasaqwhf, GFX.THRESHOLD, GFX.CONF_LEVEL, f, w_fbzzmiw_esv=w_fbzzmiw_esv)
        Gf, qipHf = build_segment_costs_with_coverage(Pf, dmkhi_f, x_imx, dmk_wqiphw=GFX.COVERAGE_TARGET, jqeuaq_dmk=GFX.COVERAGE_PENALTY_WEIGHT, H_dqvasaqwhf=H_dqvasaqwhf, wfwqi0=wfwqi0, bzzhi_eqipsv=0.0)
        fhpf_f = dp_optimal_partition(Gf, GFX.N_GROUPS, GFX.MIN_SEG_SIZE)
        Hp_ug_fspeq.append([float(H_dqvasaqwhf[qipHf[s, t]]) for s, t in fhpf_f])
    plt.figure(figsize=(7.2, 4.8))
    for sac in range(GFX.N_GROUPS):
        kqjf = [Hp[sac] for Hp in Hp_ug_fspeq]
        plt.plot(GFX.SIGMA_M_LIST, kqjf, marker='o', label=f'组{sac + 1}')
    plt.xlabel('sigma_m (logit)')
    plt.ylabel('组统一时点 T_g / 周')
    plt.title('T_g 的测量误差敏感性（固定段数，按真实分布加权）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GFX.OUT_DIR + 'sensitivity_Tg_sigma.png', dpi=160)
    plt.close()
