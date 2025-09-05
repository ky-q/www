# -*- coding: utf-8 -*-
"""
Q2 全流程（含风险分析，已消除重复 I/O 与重复构造 person-week）：
- person-week 面板 + Logit-hazard（周段 dummy + BMI）
- 输出各 BMI 组的最优检测周 τ*（最小化期望损失）
- 风险分析：
  A) 簇稳健 SE（cluster-robust by id）
  B) 校准图 + Brier 分数
  C) 成本 / 窗口权重 / 阈值 的敏感性分析
  D) Bootstrap（按个体重抽样）给 τ* 与 p(τ*) 置信区间
"""

import os, re, traceback
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 通用工具 =====================

def parse_ga_weeks_series(series: pd.Series) -> pd.Series:
    """把 '12w+3' / '12+3' / '12周3天' / '12w' / '12.5' 等转换为小数周"""
    def parse_one(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower().replace("＋", "+").replace("周", "w").replace("天", "d")
        m = re.match(r"^\s*(\d+)\s*(?:w)?\s*\+?\s*(\d+)\s*(?:d)?\s*$", s)
        if m: return float(m.group(1)) + float(m.group(2)) / 7.0
        m2 = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:w)?\s*$", s)
        if m2: return float(m2.group(1))
        try: return float(s)
        except: return np.nan
    return series.apply(parse_one)

def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def week_group_name(w: int) -> str:
    """把周映射到 5 个大区间（基准组= w_08_12，不会生成列）"""
    if 8 <= w <= 12:  return "w_08_12"
    if 13 <= w <= 16: return "w_13_16"
    if 17 <= w <= 20: return "w_17_20"
    if 21 <= w <= 24: return "w_21_24"
    return "w_25_30"

def equal_freq_binning(bmi_array: np.ndarray, K: int) -> Dict[str, Any]:
    """按等频把 BMI 分成 K 组，返回 {groups:[(lo,hi,mask),...], cuts:...}"""
    arr = pd.Series(bmi_array).dropna().sort_values().values
    qs = np.unique(np.quantile(arr, np.linspace(0, 1, K + 1)))
    groups = []
    for i in range(len(qs) - 1):
        lo, hi = float(qs[i]), float(qs[i + 1])
        mask = (bmi_array > lo) & (bmi_array <= hi)  # 左开右闭
        groups.append((lo, hi, mask))
    return {"groups": groups, "cuts": qs.tolist()}

# ===================== 数据准备（一次性） =====================

def prepare_clean_df_and_pw(
    excel_path: Optional[str] = "C:/Users/du230/Desktop/附件.xlsx",
    df_input: Optional[pd.DataFrame] = None,
    sheet_name: str = "男胎检测数据",
    col_id: str = "孕妇代码",
    col_ga: str = "检测孕周",
    col_bmi: str = "孕妇BMI",
    col_y: str = "Y染色体浓度",
    week_min: int = 8, week_max: int = 30,
    y_threshold: float = 0.04,
) -> Dict[str, Any]:
    """
    统一：读取→清洗个体层 df_clean → 构造 person-week pw。
    返回 {df_clean, pw, ids, first_hit}
    """
    # 读入
    if df_input is None:
        if excel_path is None:
            raise ValueError("请提供 excel_path 或 df_input")
        df_raw = pd.read_excel(excel_path, sheet_name=sheet_name).copy()
    else:
        df_raw = df_input.copy()

    # BMI 兜底
    if col_bmi not in df_raw.columns:
        if "身高" in df_raw.columns and "体重" in df_raw.columns:
            h = pd.to_numeric(df_raw["身高"], errors="coerce") / 100.0
            w = pd.to_numeric(df_raw["体重"], errors="coerce")
            df_raw[col_bmi] = w / (h * h)
        else:
            raise ValueError("未找到 BMI 列，也无法通过身高/体重推算。请检查列名。")

    # 清洗
    df_raw["ga"] = parse_ga_weeks_series(df_raw[col_ga])
    df_raw[col_bmi] = pd.to_numeric(df_raw[col_bmi], errors="coerce")
    df_raw[col_y] = pd.to_numeric(df_raw[col_y], errors="coerce").clip(0.0, 1.0)

    df = df_raw.dropna(subset=[col_id, "ga", col_bmi, col_y]).copy()
    df = df[(df["ga"] >= week_min) & (df["ga"] <= week_max)]
    df["id"] = df[col_id].astype(str)
    df["hit"] = (df[col_y] >= y_threshold).astype(int)

    # 每人首次达标周（向下取整）
    df_sorted = df.sort_values(["id", "ga"])
    first_hit = df_sorted[df_sorted["hit"] == 1].groupby("id")["ga"].min().apply(np.floor)

    # person-week
    ids = df_sorted["id"].unique().tolist()
    rows = []
    for pid in ids:
        fh = first_hit.get(pid, np.inf)
        for w in range(week_min, week_max + 1):
            at_risk = int(w <= fh)
            event = int(np.isfinite(fh) and (w == fh))
            rows.append((pid, w, at_risk, event))
    pw = pd.DataFrame(rows, columns=["id", "week", "at_risk", "event"])
    agg = df_sorted.groupby("id").agg({col_bmi: "first"})
    pw = pw.merge(agg, left_on="id", right_index=True, how="left")
    pw = pw[pw["at_risk"] == 1].copy()

    return {
        "df_clean": df,
        "pw": pw,
        "ids": ids,
        "first_hit": first_hit
    }

# ===================== 主流程：拟合 + 推荐 =====================

def q2_fit_and_recommend(
    # 数据源 / 复用
    excel_path: Optional[str] = None,
    df_input: Optional[pd.DataFrame] = None,
    pw_input: Optional[pd.DataFrame] = None,
    prep_kwargs: Optional[Dict[str, Any]] = None,
    # 其他参数
    sheet_name: str = "男胎检测数据",
    col_id: str = "孕妇代码",
    col_ga: str = "检测孕周",
    col_bmi: str = "孕妇BMI",
    col_y: str = "Y染色体浓度",
    week_min: int = 8, week_max: int = 30,
    K: int = 3,
    c_r: float = 1.0,
    candidate_taus: Tuple[int, int] = (10, 26),
    y_threshold: float = 0.04,
    win_cost_levels: Tuple[float, float, float] = (1.0, 3.0, 9.0),
    win_cost_scale: float = 1.0,
    out_dir: Optional[str] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """拟合 Logit-hazard → 计算各 BMI 组 τ*。"""
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # 1) 复用 df_clean / pw 或新建
    if prep_kwargs is None:
        prep_kwargs = dict(sheet_name=sheet_name, col_id=col_id, col_ga=col_ga,
                           col_bmi=col_bmi, col_y=col_y,
                           week_min=week_min, week_max=week_max, y_threshold=y_threshold)
    if (df_input is None) and (pw_input is None):
        prep = prepare_clean_df_and_pw(excel_path=excel_path, **prep_kwargs)
        df, pw = prep["df_clean"], prep["pw"]
    else:
        # 如果外部只给 df_input，我们仍可构建 pw；如果给了 pw_input 直接用之
        if pw_input is None:
            prep = prepare_clean_df_and_pw(df_input=df_input, **prep_kwargs)
            df, pw = prep["df_clean"], prep["pw"]
        else:
            # 同时也要 df_clean 以便分箱
            if df_input is None:
                prep = prepare_clean_df_and_pw(excel_path=excel_path, **prep_kwargs)
                df = prep["df_clean"]
            else:
                df = prepare_clean_df_and_pw(df_input=df_input, **prep_kwargs)["df_clean"]
            pw = pw_input.copy()

    if verbose:
        print(f"[INFO] person-week 样本量：{len(pw)}，个体数：{pw['id'].nunique()}")

    # 2) Logit-hazard 拟合
    pw["week_group"] = pw["week"].apply(week_group_name)
    week_dum = pd.get_dummies(pw["week_group"], drop_first=True).astype(float)
    bmi_series = pd.to_numeric(pw[col_bmi], errors="coerce").rename("bmi").astype(float)
    Xh = pd.concat([week_dum, bmi_series], axis=1)
    Xh = Xh.replace([np.inf, -np.inf], np.nan).astype(float)
    Xh = sm.add_constant(Xh, has_constant="add")
    yh = pd.to_numeric(pw["event"], errors="coerce").fillna(0).astype(int)
    mask = (~Xh.isna().any(axis=1))
    Xh, yh = Xh.loc[mask], yh.loc[mask]
    Xh_cols = Xh.columns.tolist()
    res_h = sm.Logit(yh, Xh).fit(disp=True, maxiter=500)

    # 3) 累计达标计算器
    def cum_hit_for_bmi(bmi_value: float) -> pd.DataFrame:
        surv = 1.0
        recs = []
        for w in range(week_min, week_max + 1):
            x = np.zeros(len(Xh_cols), dtype=float)
            if "const" in Xh_cols: x[Xh_cols.index("const")] = 1.0
            gname = week_group_name(w)
            if gname in Xh_cols: x[Xh_cols.index(gname)] = 1.0
            if "bmi" in Xh_cols:  x[Xh_cols.index("bmi")] = float(bmi_value)
            h = logistic(float(np.dot(res_h.params, x)))
            surv *= (1.0 - h)
            recs.append((w, 1.0 - surv))
        return pd.DataFrame(recs, columns=["week", "cum_hit"])

    # 成本函数与期望损失
    w_low, w_mid, w_high = win_cost_levels
    def window_cost(t: int) -> float:
        base = (w_low if t <= 12 else (w_mid if t <= 27 else w_high))
        return win_cost_scale * base
    def expected_loss(cum_hit_df: pd.DataFrame, tau: int) -> Tuple[float, float]:
        p_tau = float(cum_hit_df.loc[cum_hit_df["week"] <= tau, "cum_hit"].max())
        loss = (1.0 - p_tau) * (c_r + window_cost(tau))
        return loss, p_tau

    # 4) BMI 分组 & 推荐 τ*
    grouping = equal_freq_binning(df[col_bmi].values, K)
    tau_lo, tau_hi = candidate_taus
    rows_out = []

    # 图1：生存曲线（未达标）
    fig1 = plt.figure(figsize=(7, 5), dpi=140)
    for gidx, (lo, hi, mask_g) in enumerate(grouping["groups"], start=1):
        sub = df.loc[mask_g, col_bmi]
        if sub.empty: continue
        bmi_repr = float(np.median(sub))
        cdf = cum_hit_for_bmi(bmi_repr)

        best_tau, best_loss, best_p = None, 1e9, 0.0
        for tau in range(tau_lo, tau_hi + 1):
            loss, p_tau = expected_loss(cdf, tau)
            if loss < best_loss:
                best_tau, best_loss, best_p = tau, loss, p_tau

        rows_out.append({
            "group": f"G{gidx}",
            "bmi_interval": f"({lo:.2f}, {hi:.2f}]",
            "n": int(mask_g.sum()),
            "bmi_median": bmi_repr,
            "tau_star": best_tau,
            "p_tau": best_p,
            "expected_loss": best_loss
        })

        plt.plot(cdf["week"], 1.0 - cdf["cum_hit"], label=f"G{gidx}: BMI≈{bmi_repr:.1f}")
        plt.axvline(best_tau, linestyle="--")

    plt.xlabel("孕周（周）"); plt.ylabel("未达标概率 P(T>t)")
    plt.title("Q2：不同 BMI 组的未达标生存曲线（虚线为推荐检测周）")
    plt.legend(); plt.tight_layout()
    if save_plots:
        p1 = os.path.join(out_dir, "Q2_survival_curve.png")
        plt.savefig(p1, dpi=300); print("[保存图]", p1)
    plt.show()

    rec_table = pd.DataFrame(rows_out).sort_values("bmi_median").reset_index(drop=True)

    # 图2：期望损失柱状
    fig2 = plt.figure(figsize=(6, 4), dpi=140)
    plt.bar(rec_table["group"], rec_table["expected_loss"])
    plt.xlabel("BMI 分组"); plt.ylabel("期望损失（越低越好）")
    plt.title("Q2：各 BMI 组在 τ* 时的期望损失对比")
    plt.tight_layout()
    if save_plots:
        p2 = os.path.join(out_dir, "Q2_loss_bar.png")
        plt.savefig(p2, dpi=300); print("[保存图]", p2)
    plt.show()

    # 分组汇总表
    grouping_table = pd.DataFrame({
        "group": [f"G{i+1}" for i in range(len(grouping["groups"]))],
        "interval": [f"({lo:.2f}, {hi:.2f}]" for (lo, hi, _) in grouping["groups"]],
        "n": [int(mask.sum()) for (_, _, mask) in grouping["groups"]],
    })

    if verbose:
        print("\n[分组表]"); print(grouping_table)
        print("\n[推荐结果]"); print(rec_table)

    return {
        "res_h": res_h,
        "Xh_cols": Xh_cols,
        "grouping_table": grouping_table,
        "rec_table": rec_table,
        "aux": {
            "df": df, "pw": pw,
            "params": {
                "y_threshold": y_threshold,
                "win_cost_levels": win_cost_levels,
                "win_cost_scale": win_cost_scale,
                "c_r": c_r,
                "candidate_taus": candidate_taus,
                "week_range": (week_min, week_max),
                "K": K
            }
        }
    }

# ===================== 风险分析 A：簇稳健 SE（复用 pw） =====================

def q2_cluster_robust_se(
    df_input: Optional[pd.DataFrame] = None,
    pw_input: Optional[pd.DataFrame] = None,
    # 仅当需要从 Excel 重建时才用到以下参数
    excel_path: Optional[str] = None,
    prep_kwargs: Optional[Dict[str, Any]] = None,
) :
    """按个体聚类的稳健协方差（cluster by id），打印稳健摘要。"""
    if pw_input is None:
        if prep_kwargs is None: prep_kwargs = {}
        prep = prepare_clean_df_and_pw(excel_path=excel_path, df_input=df_input, **prep_kwargs)
        pw = prep["pw"]
    else:
        pw = pw_input.copy()

    X = pd.get_dummies(pw["week"].apply(week_group_name), drop_first=True).astype(float)
    X["bmi"] = pd.to_numeric(pw["孕妇BMI"], errors="coerce").astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(pw["event"], errors="coerce").fillna(0).astype(int)
    mask = (~X.isna().any(axis=1))
    X, y = X.loc[mask], y.loc[mask]
    groups = pw.loc[mask, "id"].values

    model = sm.Logit(y, X).fit(
        disp=False, maxiter=500,
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    print(model.summary())
    return model

# ===================== 风险分析 B：校准 + Brier（复用 df/pw） =====================

def q2_calibration_and_brier(
    df_input: Optional[pd.DataFrame] = None,
    pw_input: Optional[pd.DataFrame] = None,
    out_dir: Optional[str] = None,
    save_plot: bool = True,
    K: int = 3,
    prep_kwargs: Optional[Dict[str, Any]] = None
):
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # 复用 df / pw
    if (df_input is None) or (pw_input is None):
        if prep_kwargs is None: prep_kwargs = {}
        prep = prepare_clean_df_and_pw(df_input=df_input, **prep_kwargs)
        df = prep["df_clean"]; pw = prep["pw"]
    else:
        df = df_input.copy(); pw = pw_input.copy()

    # 拟合一次基础 Logit（非稳健）
    pw["week_group"] = pw["week"].apply(week_group_name)
    X = pd.get_dummies(pw["week_group"], drop_first=True).astype(float)
    X["bmi"] = pd.to_numeric(pw["孕妇BMI"], errors="coerce").astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(pw["event"], errors="coerce").fillna(0).astype(int)
    mask = (~X.isna().any(axis=1))
    X, y = X.loc[mask], y.loc[mask]
    model = sm.Logit(y, X).fit(disp=False, maxiter=500)
    Xcols = X.columns.tolist()

    # 画经验累计 vs 模型累计
    grouping = equal_freq_binning(df["孕妇BMI"].values, K)
    week_min, week_max = prep_kwargs.get("week_min", 8), prep_kwargs.get("week_max", 30)

    plt.figure(figsize=(8, 5), dpi=140)
    for gidx, (lo, hi, mask_g) in enumerate(grouping["groups"], start=1):
        sub_ids = df.loc[mask_g, "id"].unique()
        sub = pw[pw["id"].isin(sub_ids)].copy()

        obs = (sub.groupby("week")
               .agg(event=("event", "sum"),
                    at_risk=("event", "count"))
               .reset_index())
        obs["haz_obs"] = obs["event"] / obs["at_risk"].clip(lower=1)
        obs["cum_obs"] = 1 - (1 - obs["haz_obs"]).cumprod()

        bmi_med = float(df.loc[mask_g, "孕妇BMI"].median())
        surv = 1.0; rows = []
        for w in range(week_min, week_max + 1):
            x = np.zeros(len(Xcols))
            if "const" in Xcols: x[Xcols.index("const")] = 1.0
            gname = week_group_name(w)
            if gname in Xcols: x[Xcols.index(gname)] = 1.0
            if "bmi" in Xcols: x[Xcols.index("bmi")] = bmi_med
            h = logistic(float(np.dot(model.params, x)))
            surv *= (1 - h)
            rows.append((w, 1 - surv))
        pred = pd.DataFrame(rows, columns=["week", "cum_pred"])

        m = obs.merge(pred, on="week", how="inner")
        plt.plot(m["week"], m["cum_obs"], label=f"G{gidx} 经验累计")
        plt.plot(m["week"], m["cum_pred"], linestyle="--", label=f"G{gidx} 模型累计")

        brier = float(np.mean((m["cum_pred"] - m["cum_obs"]) ** 2))
        print(f"[G{gidx}] 校准 Brier ≈ {brier:.4f}")

    plt.xlabel("孕周（周）"); plt.ylabel("累计达标概率")
    plt.title("Q2 校准：经验累计 vs 模型累计（各 BMI 组）")
    plt.legend(); plt.tight_layout()
    if save_plot:
        p = os.path.join(out_dir, "Q2_calibration.png")
        plt.savefig(p, dpi=300); print("[保存图]", p)
    plt.show()

# ===================== 风险分析 C：敏感性分析（复用 df/pw） =====================

def q2_sensitivity(
    # 复用
    df_input: Optional[pd.DataFrame] = None,
    pw_input: Optional[pd.DataFrame] = None,
    # 扫描空间
    c_r_list: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    win_scale_list: Tuple[float, ...] = (0.5, 1.0, 1.5),
    y_thresh_list: Tuple[float, ...] = (0.035, 0.040, 0.045),
    out_csv: Optional[str] = None,
    # 其他
    fit_kwargs: Optional[Dict[str, Any]] = None
):
    """
    对 (c_r, 窗口成本缩放, 达标阈值) 做组合扫描。
    全程复用 df/pw，避免重复读写。
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    results = []
    for thr in y_thresh_list:
        for c_r in c_r_list:
            for s in win_scale_list:
                try:
                    out = q2_fit_and_recommend(
                        df_input=df_input, pw_input=pw_input,
                        prep_kwargs=dict(**fit_kwargs, y_threshold=thr),
                        c_r=c_r, y_threshold=thr, win_cost_scale=s,
                        save_plots=False, verbose=False, **fit_kwargs
                    )
                    rec = out["rec_table"].copy()
                    rec["c_r"] = c_r; rec["win_scale"] = s; rec["y_thresh"] = thr
                    results.append(rec)
                except Exception as e:
                    print("敏感性组合失败：", c_r, s, thr, e)

    if not results:
        print("敏感性分析未得到结果。")
        return None

    big = pd.concat(results, ignore_index=True)
    if out_csv is None:
        out_csv = os.path.join(fit_kwargs.get("out_dir", os.getcwd()), "Q2_sensitivity.csv")
    big.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[已导出] 敏感性结果：", out_csv)
    return big

# ===================== 风险分析 D：Bootstrap（仍需重抽样重拟合） =====================

def q2_bootstrap_tau_ci(
    excel_path: str,
    sheet_name: str = "男胎检测数据",
    id_col: str = "孕妇代码",
    B: int = 200,
    seed: int = 42,
    out_csv: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None
):
    """
    非参数 Bootstrap（按个体 id 重抽样 B 次）。
    注：这一步需要“重抽样 + 重拟合”，属于必要的重复计算。
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    rng = np.random.default_rng(seed)
    df0 = pd.read_excel(excel_path, sheet_name=sheet_name)
    ids = df0[id_col].astype(str).dropna().unique().tolist()

    # 基准一次（复用）
    base_prep = prepare_clean_df_and_pw(df_input=df0, **fit_kwargs)
    out_base = q2_fit_and_recommend(df_input=base_prep["df_clean"], pw_input=base_prep["pw"],
                                    prep_kwargs=fit_kwargs, save_plots=False, verbose=False, **fit_kwargs)
    base_rec = out_base["rec_table"].copy(); base_rec["iter"] = "base"
    rows = [base_rec]

    for b in range(B):
        try:
            boot_ids = rng.choice(ids, size=len(ids), replace=True)
            df_b = pd.concat([df0[df0[id_col].astype(str) == i] for i in boot_ids], ignore_index=True)
            prep_b = prepare_clean_df_and_pw(df_input=df_b, **fit_kwargs)
            out_b = q2_fit_and_recommend(df_input=prep_b["df_clean"], pw_input=prep_b["pw"],
                                         prep_kwargs=fit_kwargs, save_plots=False, verbose=False, **fit_kwargs)
            rec = out_b["rec_table"].copy(); rec["iter"] = b
            rows.append(rec)
        except Exception as e:
            print("bootstrap 失败 at b=", b, e)

    big = pd.concat(rows, ignore_index=True)
    ci = (big.groupby(["group", "bmi_interval"])
          .agg(tau_star_med=("tau_star", "median"),
               tau_star_lo=("tau_star", lambda x: np.quantile(x, 0.025)),
               tau_star_hi=("tau_star", lambda x: np.quantile(x, 0.975)),
               p_tau_med=("p_tau", "median"),
               p_tau_lo=("p_tau", lambda x: np.quantile(x, 0.025)),
               p_tau_hi=("p_tau", lambda x: np.quantile(x, 0.975)))
          .reset_index())

    if out_csv is None:
        out_csv = os.path.join(fit_kwargs.get("out_dir", os.getcwd()), "Q2_bootstrap_tau_ci.csv")
    ci.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[已导出] Bootstrap CI：", out_csv)
    print(ci)
    return ci

# ===================== 主函数（示例） =====================

def main():
    try:
        # 1) 路径
        excel_path = r"C:\Users\du230\Desktop\附件.xlsx"   # ←←← 改成你的路径
        out_dir    = r"C:\Users\du230\Desktop\国模"
        os.makedirs(out_dir, exist_ok=True)

        # 2) 一次性准备数据（df_clean + pw），供全局复用
        prep_kwargs = dict(sheet_name="男胎检测数据",
                           col_id="孕妇代码", col_ga="检测孕周",
                           col_bmi="孕妇BMI", col_y="Y染色体浓度",
                           week_min=8, week_max=30, y_threshold=0.04)
        prep = prepare_clean_df_and_pw(excel_path=excel_path, **prep_kwargs)
        df_clean, pw = prep["df_clean"], prep["pw"]

        # 3) 拟合 + 推荐（主结果，复用 df/pw）
        out = q2_fit_and_recommend(
            df_input=df_clean, pw_input=pw, prep_kwargs=prep_kwargs,
            K=3, c_r=1.0, candidate_taus=(10, 26),
            win_cost_levels=(1.0, 3.0, 9.0), win_cost_scale=1.0,
            out_dir=out_dir, save_plots=True, verbose=True
        )

        # 导出推荐结果
        rec = out["rec_table"]
        xlsx_path = os.path.join(out_dir, "Q2_recommendations.xlsx")
        rec.to_excel(xlsx_path, index=False)
        print("\n[已导出]", xlsx_path)
        print(rec)
        for _, r in rec.iterrows():
            print(f"组 {r['group']} {r['bmi_interval']}："
                  f"推荐周 τ*={int(r['tau_star'])}，一次达标概率≈{r['p_tau']:.2%}，期望损失≈{r['expected_loss']:.2f}")

        # 4) 风险分析 —— 全部复用 df/pw
        print("\n[簇稳健 SE 摘要]")
        _ = q2_cluster_robust_se(df_input=df_clean, pw_input=pw,
                                 prep_kwargs=prep_kwargs)

        print("\n[校准图 + Brier]")
        q2_calibration_and_brier(df_input=df_clean, pw_input=pw,
                                 out_dir=out_dir, save_plot=True, K=3,
                                 prep_kwargs=prep_kwargs)

        print("\n[敏感性分析（示例三三组合）]")
        _ = q2_sensitivity(
            df_input=df_clean, pw_input=pw,
            c_r_list=(0.5, 1.0, 1.5),
            win_scale_list=(1.0,),
            y_thresh_list=(0.04,),
            out_csv=os.path.join(out_dir, "Q2_sensitivity.csv"),
            fit_kwargs=dict(out_dir=out_dir, **prep_kwargs, K=3, candidate_taus=(10, 26))
        )

        # 如需 Bootstrap（建议先 B=50 试跑）
        # print("\n[Bootstrap CI（B=50 示例）]")
        # _ = q2_bootstrap_tau_ci(
        #     excel_path=excel_path, B=50, seed=42,
        #     out_csv=os.path.join(out_dir, "Q2_bootstrap_tau_ci.csv"),
        #     fit_kwargs=dict(out_dir=out_dir, **prep_kwargs, K=3, candidate_taus=(10, 26))
        # )

    except Exception as e:
        print("Q2 运行失败：", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()


