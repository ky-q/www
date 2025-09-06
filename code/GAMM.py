import numpy as np
import pandas as pd

from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from config.constant import CFG
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint
from tools.data_analyse import draw_q1_pics

def load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y):
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
    df["gest_weeks"] = pd.to_numeric(gest, errors="coerce")

    # 基本清洗
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df["Y_frac"], errors="coerce")
    df = df.dropna(subset=["mother_id", "gest_weeks", "BMI", "Y_frac"]).copy()

    # 因变量 logit 变换
    df["y_logit"] = logit_clip(df["Y_frac"].values, eps=1e-4)
    return df

def main(
    excel_path="./data/附件.xlsx",
    sheet_name="男胎检测数据",
    col_id="孕妇代码",
    col_ga="检测孕周",
    col_bmi="孕妇BMI",
    col_y="Y染色体浓度",
    use_tensor_interact=False,  # 是否加入“样条×BMI”的交互
    df_spline=None,  # spline数量
):
    # === 1) 读取 & 预处理 ===
    df = load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y)

    # === 2) 设计矩阵：孕周的 B 样条 + BMI (+ 可选交互) ===
    uniq = np.unique(df["gest_weeks"])
    if df_spline is None:
        k = max(4, min(6, uniq.size - 1))  # 自动限幅：通常 5~6 已足够
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

    # 非线性交互
    if use_tensor_interact:
        for c in S_train.columns:
            X[f"{c}:BMI"] = X[c] * X["BMI"]

    X = drop_near_constant_and_collinear(X)
    y = df["y_logit"].values
    groups = df["mother_id"].values
    
    # === 3) MixedLM 拟合（孕妇随机截距） ===
    md = MixedLM(endog=y, exog=X, groups=groups)
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    print(m.summary())

    # 计算伪 R^2
    r2 = 1 - np.sum((y - m.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)
    print("Pseudo R^2:", round(r2, 4))

    # BMI
    if "BMI" in X.columns:
        print("\n[Wald] H0: BMI = 0")
        bmi_res = m.wald_test("BMI = 0", use_f=True)
        print(bmi_res.summary())

    # 样条整体
    spline_cols = [c for c in X.columns if c.startswith("s")]
    print("\n[Wald] 样条整体 H0: 所有 s(gest_weeks) 系数 = 0")
    spline_test = wald_joint(spline_cols, m, X)

    # 样条×BMI 交互整体
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
        # 新增：训练数据的支撑范围
        "gest_min": float(df["gest_weeks"].min()),
        "gest_max": float(df["gest_weeks"].max()),
        "bmi_min": float(df["BMI"].min()),
        "bmi_max": float(df["BMI"].max()),
        "X": X,
        "m": m,
        "spline_test": spline_test,
        "df": df,
    }

# 全局模型变量，用于其他模块导入
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
    # 设置全局模型用于其他模块
    global_model = res
else:
    # 当作为模块被导入时，默认执行一次main并设置global_model
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