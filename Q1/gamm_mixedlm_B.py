# -*- coding: utf-8 -*-
"""
GAMM 近似：B 样条 + MixedLM 随机截距（方法 B）
- 因变量：Y染色体浓度（比例），用 logit 变换
- 固定效应：s(孕周) + BMI（可选：s(孕周)×BMI 的逐列交互）
- 随机效应：孕妇代码 的随机截距（处理同一孕妇多次检测的相关性）
- 显著性（数值 + 可视化）：
  * Wald 单项（BMI）
  * Wald 联合（样条整体、可选：样条×BMI 交互整体）
  * 系数森林图（95% CI，显著性高亮）
  * 效应曲线 + 95% 置信带（对固定效应部分）
- 输出：
  * 控制台：模型摘要、伪R²、Wald 检验
  * 文件：预测CSV、基础效应图、系数森林图、效应曲线（含置信带）
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# 将孕周转换为数值的函数（兼容 13w+0 / 13W+0 / 13w / 13W / 全角＋）
def convert_pregnancy_week(week_str: str) -> float:
    if week_str is None or (isinstance(week_str, float) and np.isnan(week_str)):
        return np.nan
    s = str(week_str).strip().lower().replace("＋", "+")
    try:
        if "w+" in s:
            w, d = s.split("w+")
            weeks = float(w)
            days = float(d) if d != "" else 0.0
        else:
            weeks = float(s.replace("w", ""))
            days = 0.0
        return weeks + days / 7.0
    except Exception:
        return np.nan


# （可选）更鲁棒版本，支持 '12周3天'、'12+3'、'12w3d'
def parse_ga_weeks_series(series: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower().replace("＋", "+").replace("周", "w").replace("天", "d")
        m = re.match(r"^\s*(\d+)\s*(?:w)?\s*\+?\s*(\d+)\s*(?:d)?\s*$", s)
        if m:
            return float(m.group(1)) + float(m.group(2)) / 7.0
        m2 = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:w)?\s*$", s)
        if m2:
            return float(m2.group(1))
        try:
            return float(s)
        except Exception:
            return np.nan

    return series.apply(parse_one)


def logit_clip(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main(
    excel_path="附件.xlsx",
    sheet_name="男胎检测数据",
    col_id="孕妇代码",
    col_ga="检测孕周",
    col_bmi="孕妇BMI",
    col_y="Y染色体浓度",
    use_tensor_interact=False,  # True：加入“样条×BMI”的逐列交互；False：只用 s(孕周)+BMI
    df_spline=None,  # None 自动选择；否则传入整数（3~8 常用）
):
    # === 1) 读取 & 预处理 ===
    df = pd.read_excel(excel_path, sheet_name=sheet_name).copy()

    # 重命名方便后续
    df = df.rename(
        columns={
            col_id: "mother_id",
            col_ga: "gest_raw",
            col_bmi: "BMI",
            col_y: "Y_frac",
        }
    )

    # 孕周
    gest = df["gest_raw"].apply(convert_pregnancy_week)
    # （若要更鲁棒，替换为：gest = parse_ga_weeks_series(df["gest_raw"])）
    df["gest_weeks"] = pd.to_numeric(gest, errors="coerce")

    # 基本清洗
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df["Y_frac"], errors="coerce")
    df = df.dropna(subset=["mother_id", "gest_weeks", "BMI", "Y_frac"]).copy()

    # 因变量 logit 变换
    df["y_logit"] = logit_clip(df["Y_frac"].values, eps=1e-4)

    # === 2) 设计矩阵：孕周的 B 样条 + BMI (+ 可选交互) ===
    uniq = np.unique(df["gest_weeks"])
    if df_spline is None:
        k = max(4, min(6, uniq.size - 1))  # 自动限幅：通常 5~6 已足够
    else:
        k = int(df_spline)
        k = max(3, min(k, max(4, uniq.size - 1)))

    # 注意：变量名不要叫 bs，避免覆盖 patsy 的 bs() 函数
    S_train = dmatrix(
        f"bs(gest_weeks, df={k}, degree=3, include_intercept=False)",
        data=df,
        return_type="dataframe",
        eval_env=1,  # 降低命名空间干扰
    )
    S_train.columns = [f"s{i+1}" for i in range(S_train.shape[1])]

    # 只用样条与 BMI（不加常数，避免与样条共线）
    X = S_train.copy()
    X["BMI"] = df["BMI"].values

    # 可选：非线性交互近似（逐列与 BMI 相乘）
    if use_tensor_interact:
        for c in S_train.columns:
            X[f"{c}:BMI"] = X[c] * X["BMI"]

    y = df["y_logit"].values
    groups = df["mother_id"].values

    # === 3) MixedLM 拟合（孕妇随机截距） ===
    md = MixedLM(endog=y, exog=X, groups=groups)
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    print(m.summary())

    # 伪 R^2（仅作为参考）
    r2 = 1 - np.sum((y - m.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)
    print("Pseudo R^2:", round(r2, 4))

    # === 4) 显著性（Wald）：BMI 单项、样条整体、（可选）交互整体 ===
    def wald_joint(names):
        names = [n for n in names if n in X.columns]
        if not names:
            print("[Info] joint set empty:", names)
            return None
        constraint = " , ".join([f"{n} = 0" for n in names])
        res = m.wald_test(constraint, use_f=True)
        print(res.summary())
        # 取出主要值供可视化注释
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

    # BMI
    bmi_test = None
    if "BMI" in X.columns:
        print("\n[Wald] H0: BMI = 0")
        bmi_res = m.wald_test("BMI = 0", use_f=True)
        print(bmi_res.summary())
        try:
            bmi_test = {
                "F": float(np.asarray(bmi_res.statistic).squeeze()),
                "p": float(np.asarray(bmi_res.pvalue).squeeze()),
            }
        except Exception:
            bmi_test = None

    # 样条整体
    spline_cols = [c for c in X.columns if c.startswith("s")]
    print("\n[Wald] 样条整体 H0: 所有 s(gest_weeks) 系数 = 0")
    spline_test = wald_joint(spline_cols)

    # （可选）交互整体
    inter_test = None
    if use_tensor_interact:
        inter_cols = [c for c in X.columns if c.startswith("s") and ":BMI" in c]
        print("\n[Wald] 样条×BMI 交互整体 H0: 交互项全部为 0")
        inter_test = wald_joint(inter_cols)

    # === 5) 可视化 1：系数森林图（固定效应，95% CI，高亮显著性） ===
    # 取固定效应（前 n_fe 个参数）
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
    # 排序：把 BMI 放最后，其余按名称
    order = [c for c in coef_df.index if c != "BMI"] + (["BMI"] if "BMI" in coef_df.index else [])
    coef_df = coef_df.loc[order]

    plt.figure(figsize=(7, 6))
    y_pos = np.arange(len(coef_df))[::-1]
    plt.errorbar(
        coef_df["coef"], y_pos, xerr=1.96 * coef_df["se"], fmt="o", capsize=3
    )
    # 竖线：0 线
    plt.axvline(0.0, linestyle="--", linewidth=1)
    # y 轴标签（显著性星号）
    labels = []
    for name, p in coef_df["p"].items():
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        labels.append(f"{name} {star}")
    plt.yticks(y_pos, labels)
    plt.xlabel("系数（logit 空间）")
    title = "固定效应系数与95%CI（MixedLM）"
    # 在标题或注释里加入样条整体的 F 与 p
    if spline_test is not None:
        title += f"\n样条整体：F={spline_test['F']:.2f}, p={spline_test['p']:.2e}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig("gamm_coef_forest.png", dpi=200)

    # === 6) 效应曲线（回到比例空间）+ 95% 置信带（固定效应） ===
    bmis = df["BMI"].values
    qs = np.nanpercentile(bmis, [25, 50, 75]).tolist()
    g_min = max(8, float(np.nanpercentile(df["gest_weeks"], 1)))
    g_max = min(30, float(np.nanpercentile(df["gest_weeks"], 99)))
    g_grid = np.linspace(g_min, g_max, 120)

    # 固定效应协方差（取参数协方差矩阵的前 n_fe×n_fe 块）
    try:
        cov_all = np.asarray(m.cov_params())
        cov_fe = cov_all[:n_fe, :n_fe]
    except Exception:
        cov_fe = None  # 部分版本可能取不到

    preds = []
    plt.figure(figsize=(7, 5))
    for b in qs:
        new = pd.DataFrame({"gest_weeks": g_grid, "BMI": b})
        S_pred = dmatrix(
            f"bs(gest_weeks, df={k}, degree=3, include_intercept=False)",
            data=new,
            return_type="dataframe",
            eval_env=1,
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

        # 线性预测及其标准误（仅固定效应部分）
        fe_vec = beta.values  # (n_fe,)
        linpred = Xg.values @ fe_vec  # (n_grid,)
        if cov_fe is not None:
            se_lin = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, cov_fe, Xg.values))
            # 95%CI on logit
            lo = linpred - 1.96 * se_lin
            hi = linpred + 1.96 * se_lin
            # 反 logit
            pred = 1 / (1 + np.exp(-linpred))
            lo_p = 1 / (1 + np.exp(-lo))
            hi_p = 1 / (1 + np.exp(-hi))
            plt.fill_between(g_grid, lo_p, hi_p, alpha=0.2)
        else:
            pred = 1 / (1 + np.exp(-linpred))
        plt.plot(g_grid, pred, label=f"BMI={b:.1f}")

        preds.append(
            pd.DataFrame({"gest_weeks": g_grid, "BMI_level": b, "pred_Y": pred})
        )

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df.to_csv("gamm_mixedlm_predictions.csv", index=False, encoding="utf-8-sig")

    plt.axhline(0.04, ls="--")
    plt.xlabel("孕周（周）")
    plt.ylabel("预测 Y 浓度（比例）")
    title2 = "GAMM 近似：孕周样条 + 孕妇随机截距（含 95% 置信带）"
    if use_tensor_interact:
        title2 += "（含 样条×BMI 交互）"
    plt.title(title2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gamm_effects_with_ci.png", dpi=200)

    # === 7) 也保留你原来的“无置信带”效果图（可选） ===
    plt.figure(figsize=(7, 5))
    for b in np.unique(pred_df["BMI_level"]):
        sub = pred_df[np.isclose(pred_df["BMI_level"], b)]
        plt.plot(sub["gest_weeks"], sub["pred_Y"], label=f"BMI={b:.1f}")
    plt.axhline(0.04, ls="--")
    plt.xlabel("孕周（周）")
    plt.ylabel("预测 Y 浓度（比例）")
    base_title = "GAMM 近似：孕周样条 + 孕妇随机截距"
    if use_tensor_interact:
        base_title += "（含 样条×BMI 交互）"
    plt.title(base_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gamm_mixedlm_effects.png", dpi=200)

    print("Saved files:")
    print("  - gamm_mixedlm_predictions.csv")
    print("  - gamm_coef_forest.png")
    print("  - gamm_effects_with_ci.png")
    print("  - gamm_mixedlm_effects.png")


if __name__ == "__main__":
    # 默认：不加交互即可（use_tensor_interact=False），先获得稳定曲线
    main(
        excel_path="./data/附件.xlsx",
        sheet_name="男胎检测数据",
        use_tensor_interact=False,  # 如需尝试交互改成 True
        df_spline=None,            # 如要手动设样条自由度，填整数（3~8）
    )
