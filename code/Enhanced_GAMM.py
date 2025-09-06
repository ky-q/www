"""
Enhanced GAMM Model with Extended Covariates
- Y染色体浓度预测的增强型GAMM模型
- 支持扩展协变量：年龄、身高、体重残差、质量指标等
- 保持与原始RAMM.py的兼容性
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint
from config.constant import CFG

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

class DataPreprocessor:
    @staticmethod
    def standardize(series):
        """标准化为Z-score"""
        if series.std() > 0:
            return (series - series.mean()) / series.std()
        return series - series.mean()

    @staticmethod
    def process_extended_covariates(df):
        """处理扩展协变量"""
        # 常见的列名
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
    """增强版数据加载函数"""
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
    # 处理扩展协变量
    df = DataPreprocessor.process_extended_covariates(df)

    # 孕周
    gest = df["gest_raw"].apply(convert_pregnancy_week)
    # （若要更鲁棒，替换为：gest = parse_ga_weeks_series(df["gest_raw"])）
    df["gest_weeks"] = pd.to_numeric(gest, errors="coerce")

    # 基本清洗
    df["mother_id"] = df["mother_id"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df["Y_frac"], errors="coerce")
    
    # 扩展协变量清洗
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = np.nan
        
    if "height" in df.columns:
        df["height"] = pd.to_numeric(df["height"], errors="coerce")
        # 保存原始height用于体重残差计算
        df["height_orig"] = df["height"].copy()
        # 标准化height (Z-score)
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
        # 转换为百万级别，避免系数过小
        df["unique_reads"] = df["unique_reads"] / 1e6
    else:
        df["unique_reads"] = np.nan
        
    if "gc_content" in df.columns:
        df["gc_content"] = pd.to_numeric(df["gc_content"], errors="coerce")
        # 标准化gc_content (Z-score)
        gc_mean = df["gc_content"].mean()
        gc_std = df["gc_content"].std()
        if gc_std > 0:
            df["gc_content"] = (df["gc_content"] - gc_mean) / gc_std
    else:
        df["gc_content"] = np.nan
    
    # 只保留必要字段的非空记录
    df = df.dropna(subset=["mother_id", "gest_weeks", "BMI", "Y_frac"]).copy()
    
    # 创建体重残差（与BMI去相关）
    if not df["weight"].isna().all() and not df["height_orig"].isna().all():
        # 预测体重（基于BMI和身高）- 使用原始身高值
        model_weight = np.polyfit(df["BMI"] * (df["height_orig"]/100)**2, df["weight"], 1)
        expected_weight = model_weight[0] * df["BMI"] * (df["height_orig"]/100)**2 + model_weight[1]
        df["weight_residual"] = df["weight"] - expected_weight
    else:
        df["weight_residual"] = np.nan

    # 因变量 logit 变换
    df["y_logit"] = logit_clip(df["Y_frac"].values, eps=1e-4)
    
    return df

def create_design_matrix(df, k, use_tensor_interact=False):
    """创建设计矩阵，包括基础特征和扩展协变量"""
    # B样条矩阵
    S_train = dmatrix(
        f"bs(gest_weeks, df={k}, degree=3, include_intercept=False)",
        data=df,
        return_type="dataframe"
    )
    S_train.columns = [f"s{i+1}" for i in range(S_train.shape[1])]

    # 基础设计矩阵
    X = S_train.copy()
    X["BMI"] = df["BMI"].values

    # BMI交互项
    if use_tensor_interact:
        for c in S_train.columns:
            X[f"{c}:BMI"] = X[c] * X["BMI"]
    
    # 扩展协变量
    # 年龄
    if "age" in df.columns and not df["age"].isna().all():
        X["age"] = df["age"].values
    
    # 身高
    if "height" in df.columns and not df["height"].isna().all():
        X["height"] = df["height"].values
    
    # 体重残差
    if "weight_residual" in df.columns and not df["weight_residual"].isna().all():
        X["weight_residual"] = df["weight_residual"].values
    
    # 质量指标
    if "unique_reads" in df.columns and not df["unique_reads"].isna().all():
        X["unique_reads"] = df["unique_reads"].values
    
    if "gc_content" in df.columns and not df["gc_content"].isna().all():
        X["gc_content"] = df["gc_content"].values

    # 删除共线性
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
    """增强版主函数"""
    # === 1) 数据预处理 ===
    df = load_data(excel_path, sheet_name, col_id, col_ga, col_bmi, col_y)

    # === 2) 确定样条数量 ===
    uniq = np.unique(df["gest_weeks"])
    if df_spline is None:
        k = max(4, min(6, uniq.size - 1))
    else:
        k = int(df_spline)
        k = max(3, min(k, max(4, uniq.size - 1)))

    # === 3) 创建设计矩阵 ===
    X = create_design_matrix(df, k, use_tensor_interact)
    y = df["y_logit"].values
    groups = df["mother_id"].values

    # === 4) 模型拟合 ===
    md = MixedLM(endog=y, exog=X, groups=groups)
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    print(m.summary())

    # === 5) 计算R² ===
    r2 = 1 - np.sum((y - m.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)
    print("Pseudo R^2:", round(r2, 4))

    # === 6) Wald检验 ===
    bmi_test = None
    if "BMI" in X.columns:
        print("\n[Wald] H0: BMI = 0")
        bmi_res = m.wald_test("BMI = 0", use_f=True)
        print(bmi_res.summary())

    # 样条整体检验
    spline_cols = [c for c in X.columns if c.startswith("s")]
    print("\n[Wald] 样条整体 H0: 所有 s(gest_weeks) 系数 = 0")
    spline_test = wald_joint(spline_cols, m, X)

    # 交互项检验
    if use_tensor_interact:
        inter_cols = [c for c in X.columns if c.endswith(":BMI")]
        print("\n[Wald] 样条×BMI 交互整体 H0: 交互项=0")
        wald_joint(inter_cols, m, X)

    # === 7) 准备模型参数 ===
    fe_names = list(X.columns)
    n_fe = len(fe_names)
    params_all = np.asarray(m.params).ravel()
    beta = pd.Series(params_all[:n_fe], index=fe_names)
    cov_all = np.asarray(m.cov_params())
    cov_fe = cov_all[:n_fe, :n_fe]

    # 为planB_constrained.py创建一个不含weight_residual和age的列表
    filtered_columns = [col for col in X.columns if col != 'weight_residual' and col != 'age']
    # 为planB_constrained.py创建一个不含weight_residual和age的beta向量
    filtered_beta = beta.loc[filtered_columns]
    # 为planB_constrained.py创建一个不含weight_residual和age的协方差矩阵
    if cov_fe is not None:
        indices = [i for i, col in enumerate(X.columns) if col in filtered_columns]
        filtered_cov_fe = cov_fe[np.ix_(indices, indices)]
    else:
        filtered_cov_fe = None
    # === 8) 返回结果 ===
    # 构建基础结果
    result = {
        "beta": filtered_beta,  # 只保留不含weight_residual和age的系数
        "cov_fe": filtered_cov_fe,  # 只保留不含weight_residual和age的协方差
        "X_columns": filtered_columns,  # 只保留不含weight_residual和age的列名
        "formula": f"bs(GA, df={k}, include_intercept=True)",
        "spline_df": k,
        "use_tensor_interact": use_tensor_interact,
        "gest_min": float(df["gest_weeks"].min()),
        "gest_max": float(df["gest_weeks"].max()),
        "bmi_min": float(df["BMI"].min()),
        "bmi_max": float(df["BMI"].max()),
        # 扩展协变量的代表值（用于预测时设置默认值）
        "default_age": None,  # 不返回age的默认值
        "default_height": 0.0,  # 因为已标准化为Z-score，默认值为0
        "default_weight_residual": None,  # 不返回weight_residual的默认值
        "default_unique_reads": float(df["unique_reads"].median()) if not df["unique_reads"].isna().all() else None,
        "default_gc_content": 0.0,  # 因为已标准化为Z-score，默认值为0
    }

    # === 9) 生成可视化 ===
    from tools.data_analyse import draw_q1_pics

    if __name__ == "__main__":
        draw_q1_pics(df, m, X, spline_test, k, use_tensor_interact, CFG.Q3PicPath)

    return result

# 全局模型变量
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
    # 当作为模块被导入时，默认执行一次main并设置global_model
    global_model = main(
        excel_path=CFG.EXCEL_PATH,
        sheet_name=CFG.SHEET_NAME,
        use_tensor_interact=True,
    )
