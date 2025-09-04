import numpy as np, pandas as pd
from patsy import dmatrix

def parse_weeks(text):
    if pd.isna(text): return np.nan
    s = str(text).lower().strip()
    if 'w+' in s:
        try:
            w, d = s.split('w+'); return float(w) + float(d)/7.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

def prepare_q1_dataset(xlsx_path: str, sheet_name: str = "男胎检测数据",
                       gc_range=(0.35,0.65), max_filtered_ratio=0.5, spline_df=5):
    """返回 (clean_df, design_X, y, meta)，其中 design_X 含样条与交互项"""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # —— 字段统一 ——
    d = raw.copy()
    d["mother_id"] = d["孕妇代码"] if "孕妇代码" in d.columns else np.arange(len(d))
    d["gest_weeks"] = (d["检测孕周"].apply(parse_weeks) if "检测孕周" in d.columns
                       else d["孕周"].apply(parse_weeks) if "孕周" in d.columns else np.nan)

    if "身高" in d.columns and "体重" in d.columns:
        h = pd.to_numeric(d["身高"], errors="coerce")
        w = pd.to_numeric(d["体重"], errors="coerce")
        with np.errstate(divide='ignore', invalid='ignore'):
            d["BMI_calc"] = w / (h/100.0)**2

    d["BMI"] = (pd.to_numeric(d["BMI"], errors="coerce")
                if "BMI" in d.columns else d.get("BMI_calc", np.nan))
    d["Y_frac"] = (pd.to_numeric(d["Y染色体浓度"], errors="coerce")
                   if "Y染色体浓度" in d.columns else np.nan)

    # 可选协变量
    rename_map = {
        "被过滤掉读段数的比例":"reads_filtered_ratio",
        "总读段数中在参考基因组上比对的比例":"map_ratio",
        "总读段数中重复读段的比例":"dup_ratio",
        "13号染色体的GC含量":"gc_13",
        "18号染色体的GC含量":"gc_18",
        "21号染色体的GC含量":"gc_21",
    }
    for k,v in rename_map.items():
        if k in d.columns:
            d[v] = pd.to_numeric(d[k], errors="coerce")

    # —— 质量/异常的温和筛选 ——
    lo, hi = gc_range
    for col in ["gc_13","gc_18","gc_21"]:
        if col in d.columns:
            d = d[(d[col].between(lo,hi)) | d[col].isna()]
    if "reads_filtered_ratio" in d.columns:
        d = d[d["reads_filtered_ratio"].fillna(0) <= max_filtered_ratio]

    # —— 关键缺失策略：删 ——
    d = d.dropna(subset=["gest_weeks","BMI","Y_frac"]).copy()

    # —— 特征构造：交互 + 样条 ——
    d["BMIxG"] = d["BMI"] * d["gest_weeks"]
    spline = dmatrix(f"bs(gest_weeks, df={spline_df}, include_intercept=False, degree=3)",
                     {"gest_weeks": d["gest_weeks"]}, return_type='dataframe')
    spline.columns = [f"spline_{i}" for i in range(spline.shape[1])]
    design_X = pd.concat([
        pd.Series(1.0, index=d.index, name="Intercept"),
        d[["BMI","gest_weeks","BMIxG"]],
        spline
    ], axis=1)

    y = d["Y_frac"].astype(float).values
    meta = {
        "n_samples": len(d),
        "cols_X": design_X.columns.tolist(),
        "note": "已完成孕周数值化、BMI计算、质量筛选、关键缺失删除、交互与样条构造"
    }
    return d, design_X, y, meta

#Step 1: 调用预处理函数
clean_df, design_X, y, meta = prepare_q1_dataset("./data/附件.xlsx")

# Step 2: 打印元信息看看
print(meta)   # 样本量、设计矩阵列名

# Step 3: 如果需要导出成 Excel
clean_df.to_excel("Q1_clean_data.xlsx", index=False)
design_X.to_excel("Q1_design_matrix.xlsx", index=False)