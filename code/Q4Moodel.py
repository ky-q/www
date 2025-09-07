import os, re, argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix
)
warnings.filterwarnings("ignore", category=UserWarning)
def find_col(df, patterns):
    colnames = df.columns
    for p in patterns:
        regex = re.compile(p, flags=re.I)
        for c in colnames:
            str_c = str(c)
            if re.search(regex, str_c):
                return c
    return None
def to_number(series):
    return pd.to_numeric(series, errors="coerce")
def clip_0_1(arr):
    arr = np.where(arr < 0, 0.0, arr)
    arr = np.where(arr > 1, 1.0, arr)
    return arr
def calibration_curve_bins(y_true, p_pred, n_bins=10):
    df = pd.DataFrame({'y': y_true, 'p': p_pred}).dropna()
    if df.empty:
        return np.array([]), np.array([]), np.array([])
    try:
        df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    except Exception:
        edges = np.linspace(0, 1, n_bins + 1)
        df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    grouped = df.groupby("bin")
    avg_p = grouped["p"].mean().values
    avg_y = grouped["y"].mean().values
    count = grouped.size().values
    return avg_p, avg_y, count
def build_dataset(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    col_Z13 = find_col(df, [r"13.*Z值", r"13号.*Z", r"Z.*13", r"chr.?13.*Z"])
    col_Z18 = find_col(df, [r"18.*Z值", r"18号.*Z", r"Z.*18", r"chr.?18.*Z"])
    col_Z21 = find_col(df, [r"21.*Z值", r"21号.*Z", r"Z.*21", r"chr.?21.*Z"])
    col_ZX  = find_col(df, [r"X.*Z值", r"X染色体.*Z", r"Z.*X", r"chr.?X.*Z"])
    col_GC13 = find_col(df, [r"13.*GC含量", r"GC.*13"])
    col_GC18 = find_col(df, [r"18.*GC含量", r"GC.*18"])
    col_GC21 = find_col(df, [r"21.*GC含量", r"GC.*21"])
    col_dup   = find_col(df, [r"重复.*比例", r"重复率", r"\bdup\b"])
    col_badr  = find_col(df, [r"被过滤.*比例", r"过滤.*比例", r"bad|filter"])
    col_BMI   = find_col(df, [r"\bBMI\b", r"体质指数"])
    col_label = find_col(df, [r"非整倍体", r"异常.*判定", r"trisomy", r"label"])
    mapping = {
        'Z13': col_Z13, 'Z18': col_Z18, 'Z21': col_Z21, 'ZX': col_ZX,
        'GC_13': col_GC13, 'GC_18': col_GC18, 'GC_21': col_GC21,
        'dup': col_dup, 'badratio': col_badr, 'BMI': col_BMI,
        'label': col_label
    }
    y_text = df[mapping['label']].astype(str).fillna("").str.upper()
    y_bin = (y_text.str.contains("T13") |
             y_text.str.contains("T18") |
             y_text.str.contains("T21")).astype(int)
    data = pd.DataFrame({
        'y': y_bin,
        'Z13': to_number(df[mapping['Z13']]) if mapping['Z13'] else np.nan,
        'Z18': to_number(df[mapping['Z18']]) if mapping['Z18'] else np.nan,
        'Z21': to_number(df[mapping['Z21']]) if mapping['Z21'] else np.nan,
        'ZX':  to_number(df[mapping['ZX']])  if mapping['ZX']  else np.nan,
        'GC_13': to_number(df[mapping['GC_13']]) if mapping['GC_13'] else np.nan,
        'GC_18': to_number(df[mapping['GC_18']]) if mapping['GC_18'] else np.nan,
        'GC_21': to_number(df[mapping['GC_21']]) if mapping['GC_21'] else np.nan,
        'dup': to_number(df[mapping['dup']]) if mapping['dup'] else np.nan,
        'badratio': to_number(df[mapping['badratio']]) if mapping['badratio'] else np.nan,
        'BMI': to_number(df[mapping['BMI']]) if mapping['BMI'] else np.nan,
    })
    data = data.dropna(subset=['Z13','Z18','Z21'], how='any').reset_index(drop=True)
    for col in list(data.columns):
        if col == 'y':
            continue
        arr = data[col].values.astype(float)
        if np.all(np.isnan(arr)):
            data.drop(columns=[col], inplace=True)
        else:
            med = np.nanmedian(arr)
            data[col] = np.where(np.isnan(arr), med, arr)
    Z_value = data[['Z13','Z18','Z21']].values
    data['s'] = (np.abs(Z_value).max(axis=1) >= 3).astype(int)
    def score_from_large_is_bad(x):
        p10 = np.nanpercentile(x, 10)
        p90 = np.nanpercentile(x, 90)
        return clip_0_1(1 - (x - p10) / (p90 - p10 + 1e-6))
    if all(name in data.columns for name in ['GC_13','GC_18','GC_21']):
        GC_mean = data[['GC_13','GC_18','GC_21']].mean(axis=1)
        GC_dev = np.abs(GC_mean - 0.5)
        gc_score = clip_0_1(1 - np.maximum(0, GC_dev - 0.10)*5)
    else:
        gc_score = np.ones(len(data)) * 0.8
    if "dup" in data.columns:
        dup_score = score_from_large_is_bad(data["dup"])
    else:
        dup_score = np.full(len(data), 0.8)
    if "badratio" in data.columns:
        badr_score = score_from_large_is_bad(data["badratio"])
    else:
        badr_score = np.full(len(data), 0.8)
    QC = 0.7 * gc_score + 0.2 * dup_score + 0.1 * badr_score
    data["QC"] = np.clip(QC, 0.0, 1.0)
    return data, mapping
def train_and_eval(data, outdir, c_fn=20.0, c_fp=1.0, c_abst=2.0, seed=42):
    os.makedirs(outdir, exist_ok=True)
    candidates = ['Z13','Z18','Z21','ZX','GC_13','GC_18','GC_21','dup','badratio','BMI']
    feat_cols = [col for col in candidates
                 if (col in data.columns) and (not np.all(np.isnan(data[col].values)))]
    X_all = data[feat_cols].fillna(0.0).values
    s_all = data['s'].values
    qc_all= data['QC'].values
    y_all = data['y'].values
    X_tr, X_te, y_tr, y_te, s_tr, s_te, qc_tr, qc_te = train_test_split(
        X_all, y_all, s_all, qc_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    X_tr, X_va, y_tr, y_va, s_tr, s_va, qc_tr, qc_va = train_test_split(
        X_tr, y_tr, s_tr, qc_tr, test_size=0.25, random_state=seed, stratify=y_tr
    )
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)
    X_tes = scaler.transform(X_te)
    base_lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=400, class_weight='balanced')
    try:
        calibrated = CalibratedClassifierCV(estimator=base_lr, method='isotonic', cv=3)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=base_lr, method='isotonic', cv=3)
    calibrated.fit(X_trs, y_tr)
    p_lr_va = calibrated.predict_proba(X_vas)[:,1]
    p_lr_te = calibrated.predict_proba(X_tes)[:,1]
    def fuse_prob(p_lr, s, qc, a0, a1):
        alpha = np.clip(a0 + a1*qc, 0.0, 1.0)
        return alpha*s + (1.0 - alpha)*p_lr
    def cost_metric(y_true, p_fused, tau_low, tau_high):
        # 将边界向外推 1 个 ULP（机器精度单位）
        hi = np.nextafter(np.float64(tau_high), np.float64(np.inf))   # 稍微抬高高阈值
        lo = np.nextafter(np.float64(tau_low),  -np.float64(np.inf))  # 稍微降低低阈值

        pred = np.full_like(y_true, fill_value=-1)
        pred[p_fused >= hi] = 1
        pred[p_fused <= lo] = 0

        FP = int(np.sum((y_true == 0) & (pred == 1)))
        FN = int(np.sum((y_true == 1) & (pred == 0)))
        N_abst = int(np.sum(pred == -1))
        N = len(y_true)
        return ((c_fn*FN + c_fp*FP + c_abst*N_abst) / max(N, 1), FP, FN, N_abst)

    a0_grid = np.linspace(0.0, 0.6, 7)
    a1_grid = np.linspace(0.0, 1.0, 6)
    tau_low_grid = np.linspace(0.05, 0.40, 18)
    tau_high_grid= np.linspace(0.60, 0.95, 18)
    best = (1e18, None)
    records = []
    for a0 in a0_grid:
        for a1 in a1_grid:
            p_va = fuse_prob(p_lr_va, s_va, qc_va, a0, a1)
            for tl in tau_low_grid:
                for th in tau_high_grid:
                    if th <= tl:
                        continue
                    c, FP, FN, N_abst = cost_metric(y_va, p_va, tl, th)
                    records.append((a0,a1,tl,th,c,FP,FN,N_abst))
                    if c < best[0]:
                        best = (c, (a0,a1,tl,th))
    best_cost, (best_a0,best_a1,best_tl,best_th) = best
    p_te = fuse_prob(p_lr_te, s_te, qc_te, best_a0, best_a1)
    test_cost, FP, FN, N_abst = cost_metric(y_te, p_te, best_tl, best_th)
    fpr, tpr, _ = roc_curve(y_te[p_te==p_te], p_te[p_te==p_te])
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    auprc = average_precision_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    pred_te = np.full_like(y_te, fill_value=-1)
    pred_te[p_te >= best_th] = 1
    pred_te[p_te <= best_tl] = 0
    mask_eval = pred_te!=-1
    cm = confusion_matrix(y_te[mask_eval], pred_te[mask_eval], labels=[0,1])
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle='--', label="随机")
    plt.xlabel("假阳性率 (False Positive Rate)")
    plt.ylabel("真阳性率 (True Positive Rate)")
    plt.title("ROC 曲线（融合概率，测试集）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Q4_s1_ROC.png"), dpi=180)
    plt.close()
    plt.figure()
    plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("召回率 (Recall)")
    plt.ylabel("精确率 (Precision)")
    plt.title("精确率-召回率曲线（融合概率，测试集）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Q4_s1_PR.png"), dpi=180)
    plt.close()
    xcal, ycal, ncal = calibration_curve_bins(y_te, p_te, n_bins=10)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle='--', label="理想校准线")
    if len(xcal) > 0:
        plt.plot(xcal, ycal, marker='o', label="分箱可靠性")
    plt.xlabel("预测概率 (融合)")
    plt.ylabel("实际阳性率")
    plt.title(f"校准曲线（Brier分数={brier:.3f}）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Q4_s1_calibration.png"), dpi=180)
    plt.close()
    p_va_best_alpha = fuse_prob(p_lr_va, s_va, qc_va, best_a0, best_a1)
    TL, TH = np.meshgrid(tau_low_grid, tau_high_grid)
    cost_grid = np.zeros_like(TL, dtype=float)
    for i in range(TL.shape[0]):
        for j in range(TL.shape[1]):
            tl = TL[i,j]; th = TH[i,j]
            if th <= tl:
                cost_grid[i,j] = np.nan
            else:
                cost_grid[i,j] = cost_metric(y_va, p_va_best_alpha, tl, th)[0]
    plt.figure()
    extent = [tau_low_grid.min(), tau_low_grid.max(), tau_high_grid.min(), tau_high_grid.max()]
    plt.imshow(np.flipud(cost_grid.T), aspect='auto', extent=extent)
    plt.scatter([best_tl], [best_th])
    plt.xlabel("低阈值 tau_low")
    plt.ylabel("高阈值 tau_high")
    plt.title("验证集代价热力图（固定最佳 a0,a1）\n圆点=选择的阈值")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Q4_s1_cost_heatmap.png"), dpi=180)
    plt.close()
    base_for_coef = LogisticRegression(penalty='l1', solver='liblinear',
                                    max_iter=400, class_weight='balanced')
    base_for_coef.fit(StandardScaler().fit_transform(X_tr), y_tr)
    coef = base_for_coef.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1]
    plt.figure(figsize=(9,4))
    plt.bar(range(len(coef)), coef[order])
    plt.xticks(range(len(coef)), [feat_cols[i] for i in order], rotation=60, ha='right')
    plt.ylabel("回归系数（标准化后）")
    plt.title("L1-逻辑回归系数（特征重要性）")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Q4_s1_coefficients.png"), dpi=180)
    plt.close()
    summary = pd.DataFrame({
        'metric': ['AUC','AUPRC','Brier','Best a0','Best a1','tau_low','tau_high','Test cost','FP','FN','Gray (abstain)','Eval samples'],
        'value':  [roc_auc, auprc, brier, best_a0, best_a1, best_tl, best_th, test_cost, FP, FN, N_abst, int(mask_eval.sum())]
    })
    summary.to_csv(os.path.join(outdir, "Q4_scheme1_test_summary.csv"), index=False)
    rec_df = pd.DataFrame(records, columns=['a0','a1','tau_low','tau_high','cost','FP','FN','N_abst'])
    rec_df.to_csv(os.path.join(outdir, "Q4_scheme1_val_gridsearch.csv"), index=False)
    return {
        'best_params': {'a0':best_a0,'a1':best_a1,'tau_low':best_tl,'tau_high':best_th,'val_cost':best_cost},
        'test_metrics': {'AUC':roc_auc,'AUPRC':auprc,'Brier':brier,'Test_cost':test_cost,'FP':FP,'FN':FN,'Gray':N_abst,'Eval_samples':int(mask_eval.sum())}
    }
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str,
                    default="./data/附件.xlsx",
                    help="输入 Excel 文件路径")
    ap.add_argument("--outdir", type=str,
                    default="./output/Q4",
                    help="输出目录")
    ap.add_argument("--c_fn", type=float, default=20.0, help="漏判成本（默认 20）")
    ap.add_argument("--c_fp", type=float, default=1.0, help="误判成本（默认 1）")
    ap.add_argument("--c_abst", type=float, default=2.0, help="灰区复检成本（默认 2）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = ap.parse_args()
    print("读取数据并构建数据集...")
    data, mapping = build_dataset(args.xlsx)
    print("列名映射：", json.dumps(mapping, ensure_ascii=False, indent=2))
    print("特征列（实际使用）：", [c for c in data.columns if c not in ['y','s','QC']])
    print("训练与评估中...")
    res = train_and_eval(
        data,
        outdir=args.outdir,
        c_fn=args.c_fn, c_fp=args.c_fp, c_abst=args.c_abst,
        seed=args.seed
    )
    print("验证最优参数：", json.dumps(res['best_params'], ensure_ascii=False, indent=2))
    print("测试集指标：", json.dumps(res['test_metrics'], ensure_ascii=False, indent=2))
    print("输出文件目录：", os.path.abspath(args.outdir))
    print("包含：Q4_s1_ROC.png, Q4_s1_PR.png, Q4_s1_calibration.png, Q4_s1_cost_heatmap.png, Q4_s1_coefficients.png,")
    print("     Q4_scheme1_test_summary.csv, Q4_scheme1_val_gridsearch.csv")
if __name__ == "__main__":
    main()
