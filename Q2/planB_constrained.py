# -*- coding: utf-8 -*-
"""
Plan B（二问，使用一问模型；按 Excel 的真实 BMI 分布分段）
----------------------------------------------------------------
- 从第一问的 global_model 取得预测器与支撑范围
- BMI 取自 Excel（默认：./data/附件.xlsx 的“男胎检测数据”），按孕妇去重
- 如样本量很大，做等频分箱压缩为最多 MAX_BMI_POINTS 个代表点，并用权重保持分布
- 段成本：最原始定义（不加波动惩罚）
- DP：固定段数（全局最优）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from gamm_mixedlm_B import global_model  # 第一问里定义好的 dict

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 配置 =================
class CFG:
    # ---------- 数据来源（用于 BMI 分布） ----------
    EXCEL_PATH = "./data/附件.xlsx"
    SHEET_NAME = "男胎检测数据"
    COL_ID = "孕妇代码"
    COL_BMI = "孕妇BMI"

    USE_EMPIRICAL_BMI = True      # True: 使用 Excel 实际分布；False: 使用等距网格
    DEDUP_BY_MOTHER = True        # 按孕妇去重（用该孕妇 BMI 的中位数）
    MAX_BMI_POINTS = 250          # 若人数过多，压缩为至多这些代表点（等频分箱），并带权重

    # ---------- 阈值与不确定性 ----------
    THRESHOLD = 0.05
    CONF_LEVEL = 0.975
    SIGMA_M_LIST = [0.0, 0.1, 0.2]
    SIGMA_M = 0.10

    # ---------- 风险分段（题意） ----------
    RISK_EARLY, RISK_MID, RISK_LATE = 1.0, 2.0, 4.0

    # ---------- 复检与成本 ----------
    RETEST_COST = 1.5           # 复检单位成本 c_retest
    VISIT_INTERVAL = 1.0        # 复检间隔（周）
    FIRST_VISIT_COST = 0.0      # 首检固定成本（可留 0）
    MAX_RETESTS = 10            # 安全上限

    # ---------- 分段设置（固定段数） ----------
    N_GROUPS = 3
    MIN_SEG_SIZE = 5

    # ---------- 下限策略（二选一；默认软下限） ----------
    HARD_FLOOR = False          # True=硬下限：不在训练下限之前搜索
    SOFT_FLOOR = True           # True=软下限：允许更早，但人为增大 SE
    EXTRAP_SIGMA_PER_WEEK = 0.45
    SOFT_BAND = 1.5
    SOFT_BETA = 4.0
    SOFT_POWER = 1.5

    # ---------- 搜索范围与步长 ----------
    T_MIN_RAW, T_MAX, STEP = 5.0, 30.0, 0.1

    # ---------- 输出 ----------
    OUT_GROUP_SUMMARY = "group_summary.csv"
    OUT_TSTAR_PNG = "t_star_vs_bmi.png"
    OUT_GROUPS_PNG = "groups_on_curve.png"
    OUT_SENS_TG_PNG = "sensitivity_Tg_sigma.png"

    USE_WAIT_PENALTY = True
    WAIT_PENALTY_PER_WEEK = 1.0   # α，和 RETEST_COST 同量纲


# =============== 工具函数 ===============
def logistic(x): return 1.0 / (1.0 + np.exp(-x))
def z_from_conf(level): return norm.ppf(level)

def piecewise_risk(t):
    if t <= 12:
        return CFG.RISK_EARLY
    elif t <= 27:
        return CFG.RISK_MID
    else:
        return CFG.RISK_LATE

def pava_monotone_increasing(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    xi, yi = x[mask], y[mask]
    n = len(yi)
    if n == 0:
        return y
    y_hat = yi.copy()
    w = np.ones_like(yi)
    i = 0
    while i < n - 1:
        if y_hat[i] <= y_hat[i + 1]:
            i += 1
        else:
            s = w[i] * y_hat[i] + w[i + 1] * y_hat[i + 1]
            ww = w[i] + w[i + 1]
            y_hat[i] = y_hat[i + 1] = s / ww
            w[i] = w[i + 1] = ww
            j = i
            while j > 0 and y_hat[j - 1] > y_hat[j]:
                s = w[j - 1] * y_hat[j - 1] + w[j] * y_hat[j]
                ww = w[j - 1] + w[j]
                y_hat[j - 1] = y_hat[j] = s / ww
                w[j - 1] = w[j] = ww
                j -= 1
            i = max(j, 0)
    out = y.copy()
    out[mask] = y_hat
    return out

def smooth_ma(y, k=5):
    k = max(1, int(k))
    if k == 1:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def precompute_tstar0(predictor, bmi, t_min, thr, conf, sigma_m, t_support_min=None):
    return np.array([
        first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                             thr, conf, sigma_m, t_support_min=t_support_min, step=CFG.STEP)
        for b in bmi
    ], dtype=float)


# =============== 直接用第一问的 GAMM 预测器 ===============
class GAMMPredictor:
    def __init__(self, beta, cov_fe, X_columns, spline_df, use_tensor_interact=False):
        self.beta = np.asarray(beta)
        self.cov_fe = np.asarray(cov_fe) if cov_fe is not None else None
        self.cols = list(X_columns)
        self.k = int(spline_df)
        self.use_tensor_interact = use_tensor_interact

    def predict_logit_and_se(self, t, b):
        from patsy import dmatrix
        import pandas as pd
        df_new = pd.DataFrame({"gest_weeks": t, "BMI": b})
        S = dmatrix(
            f"bs(gest_weeks, df={self.k}, degree=3, include_intercept=False)",
            data=df_new, return_type="dataframe", eval_env=1
        )
        S.columns = [f"s{i+1}" for i in range(S.shape[1])]
        Xg = S.copy()
        Xg["BMI"] = df_new["BMI"].values
        if self.use_tensor_interact:
            for c in S.columns:
                col = f"{c}:BMI"
                if col in self.cols:
                    Xg[col] = Xg[c] * Xg["BMI"]
        Xg = Xg.reindex(columns=self.cols, fill_value=0.0)
        eta = Xg.values @ self.beta
        if self.cov_fe is not None:
            se = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, self.cov_fe, Xg.values))
        else:
            se = np.full_like(eta, np.nan)
        return eta, se


# =============== 置信下界（带可选软下限惩罚） ===============
def _softplus(x, beta=4.0):
    return (1.0 / beta) * np.log1p(np.exp(beta * x))

def lcl_prob_from_logit(eta, se, conf, sigma_m, t=None, t_support_min=None):
    z = norm.ppf(conf)
    extra_var = 0.0
    if CFG.SOFT_FLOOR and (t is not None) and (t_support_min is not None):
        t = np.asarray(t, dtype=float)
        d = (t_support_min + CFG.SOFT_BAND) - t
        gap = _softplus(d, beta=CFG.SOFT_BETA)
        extra_sigma = CFG.EXTRAP_SIGMA_PER_WEEK * (gap ** CFG.SOFT_POWER)
        extra_var = (extra_sigma ** 2)
    s_eff = np.sqrt(se**2 + sigma_m**2 + extra_var)
    return 1.0 / (1.0 + np.exp(-(eta - z * s_eff)))


# =============== 达标时刻 & 个体损失（最原始定义） ===============
def first_hit_time_for_b(pred, b, t_min, t_max, thr, conf, sigma_m, t_support_min=None, step=0.1):
    ts = np.arange(t_min, t_max + 1e-9, step)
    bb = np.full_like(ts, b, dtype=float)
    eta, se = pred.predict_logit_and_se(ts, bb)
    lcl = lcl_prob_from_logit(eta, se, conf, sigma_m, t=ts, t_support_min=t_support_min)
    ge = np.where(lcl >= thr)[0]
    if ge.size == 0:
        return None
    j = ge[0]
    if j == 0:
        return float(ts[0])
    t0, t1 = ts[j - 1], ts[j]
    y0, y1 = lcl[j - 1], lcl[j]
    w = (thr - y0) / (y1 - y0) if y1 != y0 else 1.0
    return float(t0 + w * (t1 - t0))

def expected_hit_time(pred, b, T, thr, conf, sigma_m, t_support_min=None):
    ts = np.arange(T, CFG.T_MAX + 1e-9, CFG.STEP)
    bb = np.full_like(ts, b, dtype=float)
    eta, se = pred.predict_logit_and_se(ts, bb)
    lcl = lcl_prob_from_logit(eta, se, conf, sigma_m, t=ts, t_support_min=t_support_min)
    ge = np.where(lcl >= thr)[0]
    if ge.size == 0:
        return float(ts[-1])
    return float(ts[ge[0]])

def individual_loss(pred, b, T, thr, conf, sigma_m, cost, t_support_min=None):
    """
    个人损失（最原始）：
      1{LCL(T,b)<thr} * c_retest  +  Risk(T + Δ(b,T))
    简化实现：按复检步长，把等待时间 Δ 换算为复检次数进行计费，
            风险按真正检出时刻 piecewise_risk(t_hit) 计算。
    """
    t_hit = expected_hit_time(pred, b, T, thr, conf, sigma_m, t_support_min=t_support_min)
    gap = max(0.0, t_hit - T)
    n_retests = int(np.ceil(gap / CFG.VISIT_INTERVAL))
    n_retests = min(n_retests, CFG.MAX_RETESTS)
    retest_cost = n_retests * cost
    risk = piecewise_risk(t_hit)
    return CFG.FIRST_VISIT_COST + retest_cost + risk


# =============== 读 Excel 得到“真实 BMI 分布” ===============
def load_empirical_bmi(excel_path, sheet_name, col_id, col_bmi,
                       dedup_by_mother=True, max_points=250):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.rename(columns={col_id: "mother_id", col_bmi: "BMI"})
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df = df.dropna(subset=["mother_id", "BMI"])
    if dedup_by_mother:
        # 按孕妇去重：用该孕妇出现记录的 BMI 中位数（更鲁棒）
        bmi_series = df.groupby("mother_id")["BMI"].median().values
    else:
        bmi_series = df["BMI"].values

    bmi_series = np.asarray(bmi_series, float)
    bmi_series = bmi_series[np.isfinite(bmi_series)]
    bmi_series.sort()

    n = bmi_series.size
    if n == 0:
        raise ValueError("Excel 中未能得到有效 BMI 数据。")

    if n <= max_points:
        # 不压缩：每个孕妇一个点，权重=1
        return bmi_series, np.ones(n, dtype=float)

    # 等频分箱压缩为 <= max_points 个代表点
    # 盒边界
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


# =============== 段成本（最原始：不加任何波动惩罚） ===============
def precompute_loss_matrix(pred, bmi, Tcand, thr, conf, sigma_m, cost, t_support_min=None, w=None, tstar0=None):
    # 要求传入 tstar0 = precompute_tstar0(...)
    n, m = len(bmi), len(Tcand)
    if tstar0 is None:
        raise ValueError("precompute_loss_matrix 需要 tstar0")
    L = np.zeros((n, m))
    for i, t0 in enumerate(tstar0):
        for j, T in enumerate(Tcand):
            # 复检：当 T < t*(b)
            gap = max(0.0, t0 - T)
            n_retests = int(np.ceil(gap / CFG.VISIT_INTERVAL))
            n_retests = min(n_retests, CFG.MAX_RETESTS)
            retest_cost = n_retests * cost

            # 检出时刻与风险
            t_hit = T if T >= t0 else t0
            risk = piecewise_risk(t_hit)

            # 等待罚：当 T > t*(b)
            wait_pen = CFG.WAIT_PENALTY_PER_WEEK * max(0.0, T - t0) if CFG.USE_WAIT_PENALTY else 0.0

            L[i, j] = CFG.FIRST_VISIT_COST + retest_cost + risk + wait_pen

    if w is not None:
        L = (L.T * np.asarray(w, float)).T
    return L


def build_segment_costs(L):
    """
    C[i,j] = min_T sum_{u=i+1..j} L[u,T]
    argT[i,j] = 使段成本最小的 T 的索引
    """
    n, m = L.shape
    S = np.zeros((n + 1, m))
    for k in range(1, n + 1):
        S[k] = S[k - 1] + L[k - 1]
    C = np.full((n + 1, n + 1), np.inf)
    argT = np.full((n + 1, n + 1), -1, dtype=int)
    for i in range(n):
        for j in range(i + 1, n + 1):
            seg = S[j] - S[i]          # 该段在所有 T 下的合计损失 (m,)
            t_idx = int(np.argmin(seg))
            C[i, j] = float(seg[t_idx])
            argT[i, j] = t_idx
    return C, argT


# =============== DP：固定段数（全局最优） ===============
def dp_optimal_partition(C, K, min_seg):
    n = C.shape[0] - 1
    dp = np.full((K + 1, n + 1), 1e30)
    prev = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k * min_seg, n + 1):
            best, argi = 1e30, -1
            for i in range((k - 1) * min_seg, j - min_seg + 1):
                val = dp[k - 1, i] + C[i, j]
                if val < best:
                    best, argi = val, i
            dp[k, j] = best
            prev[k, j] = argi
    segs = []
    j = n
    for k in range(K, 0, -1):
        i = prev[k, j]
        if i < 0:
            raise RuntimeError("DP 不可行，请检查 MIN_SEG_SIZE。")
        segs.append((i, j))
        j = i
    return segs[::-1]

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
        cov = np.average(1 - need, weights=ws)                  # 首检覆盖率
        ret_rate = np.average(need, weights=ws)                 # 复检率
        nret_mean = np.average(nret, weights=ws)                # 人均复检次数
        t_mean = np.average(t_hit, weights=ws)                  # 平均检出周
        late = np.average((t_hit >= 28).astype(float), weights=ws)  # 晚期占比
        risk_mean = np.average(risk, weights=ws)                # 期望风险
        cost_mean = nret_mean * CFG.RETEST_COST + risk_mean     # 总成本（可选）

        out.append({
            "group": g, "T_g": T, "bmi_min": float(bs[0]), "bmi_max": float(bs[-1]),
            "coverage": float(cov), "retest_rate": float(ret_rate),
            "mean_retests": float(nret_mean), "mean_detect_week": float(t_mean),
            "late_share": float(late), "exp_risk": float(risk_mean),
            "exp_total_cost": float(cost_mean), "n_weight": float(w)
        })

        # overall (weighted by ws)
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


# =============== 主流程 ===============
def main():
    # 1) 预测器 & 支撑范围
    predictor = GAMMPredictor(
        beta=global_model["beta"],
        cov_fe=global_model["cov_fe"],
        X_columns=global_model["X_columns"],
        spline_df=global_model["spline_df"],
        use_tensor_interact=global_model["use_tensor_interact"],
    )

    gest_min = float(global_model.get("gest_min", 11.0))
    bmi_min = float(global_model.get("bmi_min", 18.0))
    bmi_max = float(global_model.get("bmi_max", 45.0))

    # 下限策略
    if CFG.HARD_FLOOR:
        t_min = max(CFG.T_MIN_RAW, gest_min)
        t_support_min = gest_min
    else:
        t_min = CFG.T_MIN_RAW
        t_support_min = gest_min if CFG.SOFT_FLOOR else None

    # 候选统一时点 T
    t_min_for_search = max(CFG.T_MIN_RAW, gest_min - 1.0)
    T_candidates = np.arange(t_min_for_search, CFG.T_MAX + 1e-9, CFG.STEP)

    # 2) BMI：使用 Excel 的真实分布（按孕妇去重；可压缩并带权重）
    if CFG.USE_EMPIRICAL_BMI:
        bmi_emp, w_emp = load_empirical_bmi(
            CFG.EXCEL_PATH, CFG.SHEET_NAME, CFG.COL_ID, CFG.COL_BMI,
            dedup_by_mother=CFG.DEDUP_BY_MOTHER, max_points=CFG.MAX_BMI_POINTS
        )
        bmi = bmi_emp
        w_row = w_emp
    else:
        bmi = np.linspace(bmi_min, bmi_max, 60)
        w_row = np.ones_like(bmi, dtype=float)

    # 最小段长按“代表点数量”设定
    CFG.MIN_SEG_SIZE = max(5, int(0.10 * len(bmi)))

    # 3) 段成本矩阵 & DP（固定段数）
    tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                           CFG.SIGMA_M, t_support_min=t_support_min)

    L = precompute_loss_matrix(predictor, bmi, T_candidates, CFG.THRESHOLD,
                            CFG.CONF_LEVEL, CFG.SIGMA_M, CFG.RETEST_COST,
                            t_support_min=t_support_min, w=w_row, tstar0=tstar0)

    C, argT = build_segment_costs(L)
    segments = dp_optimal_partition(C, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
    best_Ts = [float(T_candidates[argT[i, j]]) for (i, j) in segments]

    group_eval, overall = eval_schedule(
        predictor, bmi, w_row, segments, T_candidates, argT,
        CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, t_support_min=t_support_min
    )
    group_eval.to_csv("group_eval.csv", index=False, encoding="utf-8-sig")
    print("总体指标：", overall)


    # 4) 导出分组摘要
    rows = []
    for g, (i, j) in enumerate(segments, start=1):
        rows.append({
            "group": g,
            "bmi_min": float(bmi[i]),
            "bmi_max": float(bmi[j - 1]),
            "T_g": float(T_candidates[argT[i, j]]),
            "n_weight": float(np.sum(w_row[i:j]))
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(CFG.OUT_GROUP_SUMMARY, index=False, encoding="utf-8-sig")
    print("已生成:", CFG.OUT_GROUP_SUMMARY)
    print(summary)

    # 5) 图1：t*(BMI) 曲线（用连续网格画，便于阅读；不影响分段权重）
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
    if CFG.HARD_FLOOR:
        plt.axhline(t_min, ls="--", alpha=.4)
        plt.text(bmi_plot[1], t_min + 0.25, f"最早可用周={t_min:.1f}", fontsize=10)
    plt.xlabel("BMI"); plt.ylabel("最早达标周 t*")
    plt.title("达标周曲线  t*(BMI)")
    plt.grid(True, alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(CFG.OUT_TSTAR_PNG, dpi=160); plt.close()

    # 6) 图2：最优分组与统一时点（叠加 t*(b)）
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
    if CFG.HARD_FLOOR:
        plt.axhline(t_min, ls="--", alpha=.3)
    plt.xlabel("BMI"); plt.ylabel("孕周 / 周")
    plt.title("最优 BMI 分组与统一时点（固定段数，按真实分布加权）")
    plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(CFG.OUT_GROUPS_PNG, dpi=160); plt.close()

    # 7) 图3：敏感性（不同 sigma_m 下的 T_g；仍按真实分布权重）
    Tg_by_sigma = []
    for s in CFG.SIGMA_M_LIST:
        tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                                    s, t_support_min=t_support_min)
        Ls = precompute_loss_matrix(
            predictor, bmi, T_candidates,
            CFG.THRESHOLD, CFG.CONF_LEVEL, s, CFG.RETEST_COST,
            t_support_min=t_support_min, w=w_row, tstar0=tstar0
        )
        Cs, argTs = build_segment_costs(Ls)
        segs_s = dp_optimal_partition(Cs, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
        Tg_by_sigma.append([float(T_candidates[argTs[i, j]]) for (i, j) in segs_s])

    plt.figure(figsize=(7.2, 4.8))
    for idx in range(CFG.N_GROUPS):
        vals = [Tg[idx] for Tg in Tg_by_sigma]
        plt.plot(CFG.SIGMA_M_LIST, vals, marker="o", label=f"组{idx+1}")
    plt.xlabel("sigma_m (logit)"); plt.ylabel("组统一时点 T_g / 周")
    plt.title("T_g 的测量误差敏感性（固定段数，按真实分布加权）")
    plt.grid(True, alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(CFG.OUT_SENS_TG_PNG, dpi=160); plt.close()


if __name__ == "__main__":
    main()
