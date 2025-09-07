import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrix
from config.constant import CFG
from tools.data_process import piecewise_risk, pava_monotone_increasing, smooth_ma
from tools.model_utils import expected_hit_time, first_hit_time_for_b, precompute_loss_matrix, dp_optimal_partition, precompute_tstar0, build_segment_costs_simple, precompute_cover_indicator, build_segment_costs_with_coverage
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False
def wald_joint(names, m, X):
        names = [n for n in names if n in X.columns]
        if not names:
            print("[Info] joint set empty:", names)
            return None
        constraint = " , ".join([f"{n} = 0" for n in names])
        res = m.wald_test(constraint, use_f=True)
        print(res.summary())
        try:
            stat = float(np.asarray(res.statistic).squeeze())
            pval = float(np.asarray(res.pvalue).squeeze())
            df_denom, df_num = None, None
            if hasattr(res, "df_denom"):
                df_denom = res.df_denom
            if hasattr(res, "df_num"):
                df_num = res.df_num
            return {"F": stat, "p": pval, "df_num": df_num, "df_denom": df_denom}
        except Exception:
            return None
def draw_gest_distribution(excel_path, sheet_name, col_id):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    sample_counts = df[col_id].value_counts()
    print(sample_counts)
    count_distribution = sample_counts.value_counts().sort_index()
    print(count_distribution)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=count_distribution.index, y=count_distribution.values, palette="viridis")
    plt.xlabel('孕妇检验次数')
    plt.ylabel('数量')
    plt.title('孕妇检验次数分布')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(CFG.Q1PicPath + 'gest_distribution.png')
    plt.show()
def draw_q1_pics(df, m, X, spline_test, k, use_tensor_interact, save_dir):
    fe_names = list(X.columns)
    n_fe = len(fe_names)
    params_all = np.asarray(m.params).ravel()
    bse_all = np.asarray(m.bse).ravel()
    p_all = np.asarray(m.pvalues).ravel()
    beta = pd.Series(params_all[:n_fe], index=fe_names)
    se = pd.Series(bse_all[:n_fe], index=fe_names)
    pvals = pd.Series(p_all[:n_fe], index=fe_names)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    coef_df = pd.DataFrame(
        {"coef": beta, "se": se, "p": pvals, "ci_low": ci_low, "ci_high": ci_high}
    )
    order = [c for c in coef_df.index if c != "BMI"] + (["BMI"] if "BMI" in coef_df.index else [])
    coef_df = coef_df.loc[order]
    plt.figure(figsize=(7, 6))
    y_pos = np.arange(len(coef_df))[::-1]
    plt.errorbar(
        coef_df["coef"], y_pos, xerr=1.96 * coef_df["se"], fmt="o", capsize=3
    )
    plt.axvline(0.0, linestyle="--", linewidth=1)
    labels = []
    for name, p in coef_df["p"].items():
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        labels.append(f"{name} {star}")
    plt.yticks(y_pos, labels)
    plt.xlabel("系数（logit 空间）")
    title = "固定效应系数与95%CI（MixedLM）"
    if spline_test is not None:
        title += f"\n样条整体：F={spline_test['F']:.2f}, p={spline_test['p']:.2e}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir + "gamm_coef_forest.png", dpi=200)
    bmis = df["BMI"].values
    qs = np.nanpercentile(bmis, [25, 50, 75]).tolist()
    g_min = max(8, float(np.nanpercentile(df["gest_weeks"], 1)))
    g_max = min(30, float(np.nanpercentile(df["gest_weeks"], 99)))
    g_grid = np.linspace(g_min, g_max, 120)
    try:
        cov_all = np.asarray(m.cov_params())
        cov_fe = cov_all[:n_fe, :n_fe]
    except Exception:
        cov_fe = None
    plt.figure(figsize=(7, 5))
    for b in qs:
        new = pd.DataFrame({"gest_weeks": g_grid, "BMI": b})
        S_pred = dmatrix(
            f"bs(gest_weeks, df={k}, degree=3, include_intercept=False)",
            data=new,
            return_type="dataframe"
        )
        S_pred.columns = [f"s{i+1}" for i in range(S_pred.shape[1])]
        Xg = S_pred.copy()
        Xg["BMI"] = b
        if use_tensor_interact:
            for c in S_pred.columns:
                col = f"{c}:BMI"
                if col in X.columns:
                    Xg[col] = Xg[c] * Xg["BMI"]
        Xg = Xg.reindex(columns=X.columns, fill_value=0.0)
        fe_vec = beta.values
        linpred = Xg.values @ fe_vec
        if cov_fe is not None:
            se_lin = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, cov_fe, Xg.values))
            lo = linpred - 1.96 * se_lin
            hi = linpred + 1.96 * se_lin
            pred = 1 / (1 + np.exp(-linpred))
            lo_p = 1 / (1 + np.exp(-lo))
            hi_p = 1 / (1 + np.exp(-hi))
            plt.fill_between(g_grid, lo_p, hi_p, alpha=0.2)
        else:
            pred = 1 / (1 + np.exp(-linpred))
        plt.plot(g_grid, pred, label=f"BMI={b:.1f}")
    plt.axhline(0.04, ls="--")
    plt.xlabel("孕周（周）")
    plt.ylabel("预测 Y 浓度（比例）")
    title2 = "GAMM 近似：孕周样条 + 孕妇随机截距（含 95% 置信带）"
    if use_tensor_interact:
        title2 += "（含 样条×BMI 交互）"
    plt.title(title2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + "gamm_effects_with_ci.png", dpi=200)
    g_grid_3d = np.linspace(max(8.0, df["gest_weeks"].min()),
                         min(30.0, df["gest_weeks"].max()), 46)
    b_grid_3d = np.linspace(df["BMI"].min(), df["BMI"].max(), 41)
    G, B = np.meshgrid(g_grid_3d, b_grid_3d, indexing="ij")
    grid_df = pd.DataFrame({"gest_week": G.ravel(), "BMI": B.ravel()})
    S_pred = dmatrix(
        f"bs(gest_week, df={k}, degree=3, include_intercept=False)",
        data=grid_df, return_type="dataframe"
    )
    S_pred.columns = [f"s{i+1}" for i in range(S_pred.shape[1])]
    Xg = S_pred.copy()
    Xg["BMI"] = grid_df["BMI"].values
    if use_tensor_interact:
        for c in S_pred.columns:
            col = f"{c}:BMI"
            if col in X.columns:
                Xg[col] = Xg[c] * Xg["BMI"]
    Xg = Xg.reindex(columns=X.columns, fill_value=0.0)
    fe_vec = beta.values
    eta_logit = Xg.values @ fe_vec
    Y_pred = 1 / (1 + np.exp(-eta_logit))
    Z = Y_pred.reshape(G.shape)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(G, B, Z, cmap="viridis", alpha=0.7)
    ax.scatter(df["gest_weeks"], df["BMI"], df["Y_frac"],
               color="r", alpha=0.4, s=15, label="observed")
    ax.set_xlabel("孕周 (weeks)")
    ax.set_ylabel("BMI")
    ax.set_zlabel("Y 浓度 (proportion)")
    ax.set_title("GAMM 拟合的孕周-BMI-Y 浓度三维曲面")
    plt.tight_layout()
    plt.savefig(save_dir + "gamm_3d_surface.png", dpi=200)
    plt.show()
def load_empirical_bmi(excel_path, sheet_name, col_id, col_bmi, max_points=250):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.rename(columns={col_id: "mother_id", col_bmi: "BMI"})
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df = df.dropna(subset=["mother_id", "BMI"])
    bmi_series = df.groupby("mother_id")["BMI"].median().values
    bmi_series = np.asarray(bmi_series, float)
    bmi_series = bmi_series[np.isfinite(bmi_series)]
    bmi_series.sort()
    n = bmi_series.size
    if n == 0:
        raise ValueError("Excel 中未能得到有效 BMI 数据。")
    if n <= max_points:
        return bmi_series, np.ones(n, dtype=float)
    edges = np.quantile(bmi_series, np.linspace(0, 1, max_points + 1))
    reps = []
    wts = []
    for k in range(max_points):
        lo, hi = edges[k], edges[k + 1]
        if k < max_points - 1:
            mask = (bmi_series >= lo) & (bmi_series < hi)
        else:
            mask = (bmi_series >= lo) & (bmi_series <= hi)
        arr = bmi_series[mask]
        if arr.size == 0:
            continue
        reps.append(np.median(arr))
        wts.append(float(arr.size))
    reps = np.asarray(reps, float)
    wts = np.asarray(wts, float)
    order = np.argsort(reps)
    return reps[order], wts[order]
def eval_schedule(predictor, bmi, w_row, segments, T_candidates, argT,
                  thr, conf, sigma_m, t_support_min=None):
    def hit_time(b, T):
        return expected_hit_time(predictor, b, T, thr, conf, sigma_m, t_support_min=t_support_min)
    out = []
    w_all = 0.0; cov_all = ret_rate_all = nrt_all = 0.0
    risk_all = 0.0; late_all = 0.0; tbar_w = 0.0
    for g, (i, j) in enumerate(segments, start=1):
        T = float(T_candidates[argT[i, j]])
        bs = bmi[i:j]; ws = w_row[i:j]
        t_hit = np.array([hit_time(b, T) for b in bs])
        need = (t_hit > T).astype(float)
        nret = np.ceil(np.maximum(0.0, t_hit - T) / CFG.VISIT_INTERVAL)
        nret = np.minimum(nret, CFG.MAX_RETESTS)
        risk = np.array([piecewise_risk(t) for t in t_hit])
        w = ws.sum()
        cov = np.average(1 - need, weights=ws)
        ret_rate = np.average(need, weights=ws)
        nret_mean = np.average(nret, weights=ws)
        t_mean = np.average(t_hit, weights=ws)
        late = np.average((t_hit >= 28).astype(float), weights=ws)
        risk_mean = np.average(risk, weights=ws)
        cost_mean = nret_mean * CFG.RETEST_COST + risk_mean
        out.append({
            "group": g, "T_g": T, "bmi_min": float(bs[0]), "bmi_max": float(bs[-1]),
            "coverage": float(cov), "retest_rate": float(ret_rate),
            "mean_retests": float(nret_mean), "mean_detect_week": float(t_mean),
            "late_share": float(late), "exp_risk": float(risk_mean),
            "exp_total_cost": float(cost_mean), "n_weight": float(w)
        })
        w_all += w
        cov_all += cov * w; ret_rate_all += ret_rate * w; nrt_all += nret_mean * w
        risk_all += risk_mean * w; late_all += late * w; tbar_w += t_mean * w
    overall = {
        "coverage": cov_all / w_all,
        "retest_rate": ret_rate_all / w_all,
        "mean_retests": nrt_all / w_all,
        "mean_detect_week": tbar_w / w_all,
        "late_share": late_all / w_all,
        "exp_risk": risk_all / w_all
    }
    return pd.DataFrame(out), overall
def draw_q2_pics(bmi, predictor, t_min, t_support_min, segments, best_Ts, T_candidates, w_row):
    bmi_plot = np.linspace(min(bmi), max(bmi), 200)
    plt.figure(figsize=(7.6, 5.2))
    for s in CFG.SIGMA_M_LIST:
        t_star = [
            first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                                 CFG.THRESHOLD, CFG.CONF_LEVEL, s,
                                 t_support_min=t_support_min, step=CFG.STEP)
            for b in bmi_plot
        ]
        t_star = np.array([np.nan if v is None else float(v) for v in t_star])
        t_star = pava_monotone_increasing(bmi_plot, t_star)
        t_star = smooth_ma(t_star, k=5)
        plt.plot(bmi_plot, t_star, label=f"sigma_m={s}")
    plt.xlabel("BMI"); plt.ylabel("最早达标周 t*")
    plt.title("达标周曲线  t*(BMI)")
    plt.grid(True, alpha=.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(CFG.Q2PicPath + CFG.OUT_TSTAR_PNG, dpi=160); plt.close()
    plt.figure(figsize=(7.8, 5.3))
    t_star = [
        first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                             CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M,
                             t_support_min=t_support_min, step=CFG.STEP)
        for b in bmi_plot
    ]
    plt.plot(bmi_plot, t_star, label="t*(b)", lw=2)
    for (i, j), Tg in zip(segments, best_Ts):
        plt.axvspan(bmi[i], bmi[j - 1], alpha=0.08)
        plt.hlines(Tg, bmi[i], bmi[j - 1], linestyles="dashed")
        plt.text((bmi[i] + bmi[j - 1]) / 2, Tg + 0.25, f"T={Tg:.1f}", ha="center")
    plt.xlabel("BMI"); plt.ylabel("孕周 / 周")
    plt.title("最优 BMI 分组与统一时点（固定段数，按真实分布加权）")
    plt.legend();
    plt.grid(True, alpha=.3)
    plt.tight_layout();
    plt.savefig(CFG.Q2PicPath + CFG.OUT_GROUPS_PNG, dpi=160); plt.close()
    Tg_by_sigma = []
    for s in CFG.SIGMA_M_LIST:
        tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                                    s, t_support_min=t_support_min)
        Ls = precompute_loss_matrix(bmi, T_candidates, CFG.RETEST_COST, w=w_row, tstar0=tstar0
        )
        Cs, argTs = build_segment_costs_simple(Ls)
        segs_s = dp_optimal_partition(Cs, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
        Tg_by_sigma.append([float(T_candidates[argTs[i, j]]) for (i, j) in segs_s])
    plt.figure(figsize=(7.2, 4.8))
    for idx in range(CFG.N_GROUPS):
        vals = [Tg[idx] for Tg in Tg_by_sigma]
        plt.plot(CFG.SIGMA_M_LIST, vals, marker="o", label=f"组{idx+1}")
    plt.xlabel("sigma_m (logit)"); plt.ylabel("组统一时点 T_g / 周")
    plt.title("T_g 的测量误差敏感性（固定段数，按真实分布加权）")
    plt.grid(True, alpha=.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(CFG.Q2PicPath + CFG.OUT_SENS_TG_PNG, dpi=160); plt.close()
def draw_q3_pics(bmi, predictor, t_min, t_support_min, segments, best_Ts, T_candidates, w_row, CFG):
    bmi_plot = np.linspace(min(bmi), max(bmi), 200)
    plt.figure(figsize=(7.6, 5.2))
    for s in CFG.SIGMA_M_LIST:
        t_star = [
            first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                                 CFG.THRESHOLD, CFG.CONF_LEVEL, s,
                                 t_support_min=t_support_min, step=CFG.STEP)
            for b in bmi_plot
        ]
        t_star = np.array([np.nan if v is None else float(v) for v in t_star])
        t_star = pava_monotone_increasing(bmi_plot, t_star)
        t_star = smooth_ma(t_star, k=5)
        plt.plot(bmi_plot, t_star, label=f"sigma_m={s}")
    plt.xlabel("BMI"); plt.ylabel("最早达标周 t*")
    plt.title("达标周曲线  t*(BMI)")
    plt.grid(True, alpha=.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(CFG.OUT_DIR + "t_star_vs_bmi.png", dpi=160);
    plt.figure(figsize=(7.8, 5.3))
    t_star = [
        first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                             CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M,
                             t_support_min=t_support_min, step=CFG.STEP)
        for b in bmi_plot
    ]
    plt.plot(bmi_plot, t_star, label="t*(b)", lw=2)
    for (i, j), Tg in zip(segments, best_Ts):
        plt.axvspan(bmi[i], bmi[j - 1], alpha=0.08)
        plt.hlines(Tg, bmi[i], bmi[j - 1], linestyles="dashed")
        plt.text((bmi[i] + bmi[j - 1]) / 2, Tg + 0.25, f"T={Tg:.1f}", ha="center")
    plt.xlabel("BMI"); plt.ylabel("孕周 / 周")
    plt.title("最优 BMI 分组与统一时点（固定段数，按真实分布加权）")
    plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout();
    plt.savefig(CFG.OUT_DIR + "groups_on_curve.png", dpi=160);
    plt.close()
    Tg_by_sigma = []
    for s in CFG.SIGMA_M_LIST:
        tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                                    s, t_support_min=t_support_min)
        Ls = precompute_loss_matrix(bmi, T_candidates, CFG.RETEST_COST, w=w_row, tstar0=tstar0)
        cover_s = precompute_cover_indicator(
            predictor, bmi, T_candidates,
            CFG.THRESHOLD, CFG.CONF_LEVEL, s,
            t_support_min=t_support_min
        )
        Cs, argTs = build_segment_costs_with_coverage(
            Ls, cover_s, w_row,
            cov_target=CFG.COVERAGE_TARGET,
            lambda_cov=CFG.COVERAGE_PENALTY_WEIGHT,
            T_candidates=T_candidates,
            tstar0=tstar0,
            upper_margin=0.0
        )
        segs_s = dp_optimal_partition(Cs, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
        Tg_by_sigma.append([float(T_candidates[argTs[i, j]]) for (i, j) in segs_s])
    plt.figure(figsize=(7.2, 4.8))
    for idx in range(CFG.N_GROUPS):
        vals = [Tg[idx] for Tg in Tg_by_sigma]
        plt.plot(CFG.SIGMA_M_LIST, vals, marker="o", label=f"组{idx+1}")
    plt.xlabel("sigma_m (logit)"); plt.ylabel("组统一时点 T_g / 周")
    plt.title("T_g 的测量误差敏感性（固定段数，按真实分布加权）")
    plt.grid(True, alpha=.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(CFG.OUT_DIR + "sensitivity_Tg_sigma.png", dpi=160);
    plt.close()
