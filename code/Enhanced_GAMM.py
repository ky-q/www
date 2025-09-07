import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint
from config.constant import CFG
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False
def process_extended_covariates(df):
    if "年龄" in df.columns:
        df.rename(columns={"年龄": "age"}, inplace=True)
    if "身高" in df.columns:
        df.rename(columns={"身高": "height"}, inplace=True)
    if "体重" in df.columns:
        df.rename(columns={"体重": "weight"}, inplace=True)
    if "唯一比对的读段数" in df.columns:
        df.rename(columns={"唯一比对的读段数": "unique_reads"}, inplace=True)
    if "GC含量" in df.columns:
        df.rename(columns={"GC含量": "gc_content"}, inplace=True)
    return df
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
    df = process_extended_covariates(df)
    gest = df["gest_raw"].apply(convert_pregnancy_week)
    df["gest_weeks"] = pd.to_numeric(gest, errors="coerce")
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df["Y_frac"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = np.nan
    if "height" in df.columns:
        df["height"] = pd.to_numeric(df["height"], errors="coerce")
        df["height_orig"] = df["height"].copy()
        height_mean = df["height"].mean()
        height_std = df["height"].std()
        if height_std > 0:
            df["height"] = (df["height"] - height_mean) / height_std
    else:
        df["height"] = np.nan
        df["height_orig"] = np.nan
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    else:
        df["weight"] = np.nan
    if "unique_reads" in df.columns:
        df["unique_reads"] = pd.to_numeric(df["unique_reads"], errors="coerce")
        df["unique_reads"] = df["unique_reads"] / 1e6
    else:
        df["unique_reads"] = np.nan
    if "gc_content" in df.columns:
        df["gc_content"] = pd.to_numeric(df["gc_content"], errors="coerce")
        gc_mean = df["gc_content"].mean()
        gc_std = df["gc_content"].std()
        if gc_std > 0:
            df["gc_content"] = (df["gc_content"] - gc_mean) / gc_std
    else:
        df["gc_content"] = np.nan
    df = df.dropna(subset=["mother_id", "gest_weeks", "BMI", "Y_frac"]).copy()
    if not df["weight"].isna().all() and not df["height_orig"].isna().all():
        model_weight = np.polyfit(df["BMI"] * (df["height_orig"]/100)**2, df["weight"], 1)
        expected_weight = model_weight[0] * df["BMI"] * (df["height_orig"]/100)**2 + model_weight[1]
        df["weight_residual"] = df["weight"] - expected_weight
    else:
        df["weight_residual"] = np.nan
    df["y_logit"] = logit_clip(df["Y_frac"].values, eps=1e-4)
    return df
def create_design_matrix(df, k, use_tensor_interact=False):
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
    if "age" in df.columns and not df["age"].isna().all():
        X["age"] = df["age"].values
    if "height" in df.columns and not df["height"].isna().all():
        X["height"] = df["height"].values
    if "weight_residual" in df.columns and not df["weight_residual"].isna().all():
        X["weight_residual"] = df["weight_residual"].values
    if "unique_reads" in df.columns and not df["unique_reads"].isna().all():
        X["unique_reads"] = df["unique_reads"].values
    if "gc_content" in df.columns and not df["gc_content"].isna().all():
        X["gc_content"] = df["gc_content"].values
    X = drop_near_constant_and_collinear(X)
    return X
def main(
    excel_path=CFG.EXCEL_PATH,
    sheet_name="男胎检测数据",
    col_id="孕妇代码",
    col_ga="检测孕周",
    col_bmi="孕妇BMI",
    col_y="Y染色体浓度",
    use_tensor_interact=False,
    df_spline=None,
):
    df = load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y)
    uniq = np.unique(df["gest_weeks"])
    if df_spline is None:
        k = max(4, min(6, uniq.size - 1))
    else:
        k = int(df_spline)
        k = max(3, min(k, max(4, uniq.size - 1)))
    X = create_design_matrix(df, k, use_tensor_interact)
    y = df["y_logit"].values
    groups = df["mother_id"].values
    md = MixedLM(endog=y, exog=X, groups=groups)
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    print(m.summary())
    r2 = 1 - np.sum((y - m.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)
    print("Pseudo R^2:", round(r2, 4))
    bmi_test = None
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
    filtered_columns = [col for col in X.columns if col != 'weight_residual' and col != 'age']
    filtered_beta = beta.loc[filtered_columns]
    if cov_fe is not None:
        indices = [i for i, col in enumerate(X.columns) if col in filtered_columns]
        filtered_cov_fe = cov_fe[np.ix_(indices, indices)]
    else:
        filtered_cov_fe = None
    result = {
        "beta": filtered_beta,
        "cov_fe": filtered_cov_fe,
        "X_columns": filtered_columns,
        "formula": f"bs(GA, df={k}, include_intercept=True)",
        "spline_df": k,
        "use_tensor_interact": use_tensor_interact,
        "gest_min": float(df["gest_weeks"].min()),
        "gest_max": float(df["gest_weeks"].max()),
        "bmi_min": float(df["BMI"].min()),
        "bmi_max": float(df["BMI"].max()),
        "default_age": None,
        "default_height": 0.0,
        "default_weight_residual": None,
        "default_unique_reads": float(df["unique_reads"].median()) if not df["unique_reads"].isna().all() else None,
        "default_gc_content": 0.0,
    }
    from tools.data_analyse import draw_q1_pics
    if __name__ == "__main__":
        draw_q1_pics(df, m, X, spline_test, k, use_tensor_interact, CFG.Q3PicPath)
    return result
global_model = None
if __name__ == "__main__":
    global_model = main(
        excel_path=CFG.EXCEL_PATH,
        sheet_name=CFG.SHEET_NAME,
        col_id="孕妇代码",
        col_ga="检测孕周",
        col_bmi="孕妇BMI",
        col_y="Y染色体浓度",
        use_tensor_interact=True,
        df_spline=None,
    )
else:
    global_model = main(
        excel_path=CFG.EXCEL_PATH,
        sheet_name=CFG.SHEET_NAME,
        use_tensor_interact=True,
    )
