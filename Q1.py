import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False    # 图片显示中文
from patsy import dmatrix

FILE_PATH = "./data/附件.xlsx"

# 把检测孕周的带w的值 转化成浮点值
def parse_weeks(text):
    if pd.isna(text): return np.nan
    s = str(text).lower().strip()
    if 'w+' in s:
        try:
            w, d = s.split('w+'); return float(w) + float(d)/7.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

# 统一列名
def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
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
    return d

# 读取男胎数据
male_raw = pd.read_excel(FILE_PATH, sheet_name="男胎检测数据")
male = unify_columns(male_raw).dropna(subset=["gest_weeks","BMI","Y_frac"]).copy()

# B样条基函数
spline = dmatrix("bs(gest_weeks, df=5, include_intercept=False, degree=3)",
                 {"gest_weeks": male["gest_weeks"]}, return_type='dataframe')
X = pd.concat([
    pd.Series(1.0, index=male.index, name="Intercept"),
    male[["BMI","gest_weeks"]],
    (male["BMI"]*male["gest_weeks"]).rename("BMIxG"),
    spline
], axis=1)

# 稳健方差 用来OLS拟合
y = male["Y_frac"].values
ols = sm.OLS(y, X.values).fit(cov_type="HC3")

# 打印t检验结果
print("\n=========== T检验结果 ===========")
print("模型系数的t检验结果：")
print(ols.summary().tables[1])  # 显示系数、标准误、t值和p值
print("\n显著性水平：")
print("* p<0.05: 显著")
print("** p<0.01: 非常显著")
print("*** p<0.001: 极其显著")
print("================================\n")

# 构建预测函数（基于OLS系数）返回预测的Y浓度
coefs = dict(zip(X.columns, ols.params))
def mu_hat(g: float, b: float) -> float:
    sp = dmatrix("bs(x, df=5, include_intercept=False, degree=3)", {"x":[g]},
                 return_type='dataframe')
    val = coefs.get("Intercept",0.0)
    val += coefs.get("BMI",0.0)*b
    val += coefs.get("gest_weeks",0.0)*g
    val += coefs.get("BMIxG",0.0)*(b*g)
    for c in sp.columns: val += coefs.get(c,0.0)*sp[c].values[0]
    return float(val)

# 图1：不同BMI分位的拟合曲线和4%的阈值线
bmis = male["BMI"].values
qs = np.nanpercentile(bmis, [25,50,75]).tolist() if len(bmis)>0 else [22,28,34]
g_grid = np.linspace(max(8, float(np.nanpercentile(male["gest_weeks"], 1))),
                     min(30, float(np.nanpercentile(male["gest_weeks"], 99))), 120)

plt.figure()
for b in qs:
    y_pred = [mu_hat(g,b) for g in g_grid]
    plt.plot(g_grid, y_pred, label=f"BMI={round(b,1)}")
plt.axhline(0.04, linestyle="--")
plt.xlabel("孕周（周）"); plt.ylabel("Y 浓度")
plt.title("Q1：不同 BMI 分位的 Y~孕周 拟合曲线（OLS+样条，阈值=4%）")
plt.legend(); plt.tight_layout()
plt.savefig("Q1_Y_vs_g.png", dpi=200)

# 图2：原本数据和预测的BMI中位值的拟合曲线
plt.figure()
plt.scatter(male["gest_weeks"].values, male["Y_frac"].values, alpha=0.35)
b_med = float(np.nanmedian(bmis)) if len(bmis)>0 else 28.0
y_line = [mu_hat(g, b_med) for g in g_grid]
plt.plot(g_grid, y_line, label=f"拟合曲线@BMI中位 {b_med:.1f}")
plt.axhline(0.04, linestyle="--")
plt.xlabel("孕周（周）"); plt.ylabel("Y 浓度")
plt.title("Q1：原始散点 + 拟合曲线（OLS+样条）")
plt.legend(); plt.tight_layout()
plt.savefig("Q1_scatter_fit.png", dpi=200)
