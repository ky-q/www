import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from config.constant import CFG
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint, draw_gest_distribution, draw_q1_pics
def load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y):
    df = pd.read_excel(excel_path, sheet_name=sheet_name).copy()
    df = df.rename(
        columns={
            col_id: "mother_id",
            col_ga: "gest_raw",
            col_bmi: "BMI",
            col_y: "Y_frac",
        }
    )
    gest = df["gest_raw"].apply(convert_pregnancy_week)
    df["gest_weeks"] = pd.to_numeric(gest, errors="coerce")
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df["Y_frac"], errors="coerce")
    df = df.dropna(subset=["mother_id", "gest_weeks", "BMI", "Y_frac"]).copy()
    df["y_logit"] = logit_clip(df["Y_frac"].values, eps=1e-4)
    return df
def main(
    excel_path="./data/附件.xlsx",
    sheet_name="男胎检测数据",
    col_id="孕妇代码",
    col_ga="检测孕周",
    col_bmi="孕妇BMI",
    col_y="Y染色体浓度",
    use_tensor_interact=False,
    df_spline=None,
):
    df = load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y)
    draw_gest_distribution(excel_path, sheet_name, col_id)
    uniq = np.unique(df["gest_weeks"])
    if df_spline is None:
        k = max(4, min(6, uniq.size - 1))
    else:
        k = int(df_spline)
        k = max(3, min(k, max(4, uniq.size - 1)))
    S_train = dmatrix(
        f"bs(gest_weeks, df={k}, degree=3, include_intercept=False)",
        data=df,
        return_type="dataframe"
    )
    S_train.columns = [f"s{i+1}" for i in range(S_train.shape[1])]
    X = S_train.copy()
    X["BMI"] = df["BMI"].values
    if use_tensor_interact:
        for c in S_train.columns:
            X[f"{c}:BMI"] = X[c] * X["BMI"]
    X = drop_near_constant_and_collinear(X)
    y = df["y_logit"].values
    groups = df["mother_id"].values
    md = MixedLM(endog=y, exog=X, groups=groups)
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    print(m.summary())
    r2 = 1 - np.sum((y - m.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)
    print("Pseudo R^2:", round(r2, 4))
    if "BMI" in X.columns:
        print("\n[Wald] H0: BMI = 0")
        bmi_res = m.wald_test("BMI = 0", use_f=True)
        print(bmi_res.summary())
    spline_cols = [c for c in X.columns if c.startswith("s")]
    print("\n[Wald] 样条整体 H0: 所有 s(gest_weeks) 系数 = 0")
    spline_test = wald_joint(spline_cols, m, X)
    if use_tensor_interact:
        inter_cols = [c for c in X.columns if c.endswith(":BMI")]
        print("\n[Wald] 样条×BMI 交互整体 H0: 交互项=0")
        wald_joint(inter_cols, m, X)
    fe_names = list(X.columns)
    n_fe = len(fe_names)
    params_all = np.asarray(m.params).ravel()
    beta = pd.Series(params_all[:n_fe], index=fe_names)
    cov_all = np.asarray(m.cov_params())
    cov_fe = cov_all[:n_fe, :n_fe]
    if __name__ == "__main__":
        draw_q1_pics(df, m, X, spline_test, k, use_tensor_interact, CFG.Q1PicPath)
    return {
        "beta": beta,
        "cov_fe": cov_fe,
        "X_columns": X.columns,
        "spline_df": k,
        "use_tensor_interact": use_tensor_interact,
        "gest_min": float(df["gest_weeks"].min()),
        "gest_max": float(df["gest_weeks"].max()),
        "bmi_min": float(df["BMI"].min()),
        "bmi_max": float(df["BMI"].max()),
        "X": X,
        "m": m,
        "spline_test": spline_test,
        "df": df,
    }
global_model = None
if __name__ == "__main__":
    res = main(
        excel_path=CFG.EXCEL_PATH,
        sheet_name=CFG.SHEET_NAME,
        col_id="孕妇代码",
        col_ga="检测孕周",
        col_bmi="孕妇BMI",
        col_y="Y染色体浓度",
        use_tensor_interact=False,
        df_spline=None,
    )
    global_model = res
else:
    global_model = main(
        excel_path=CFG.EXCEL_PATH,
        sheet_name=CFG.SHEET_NAME,
        col_id="孕妇代码",
        col_ga="检测孕周",
        col_bmi="孕妇BMI",
        col_y="Y染色体浓度",
        use_tensor_interact=False,
        df_spline=None,
    )
