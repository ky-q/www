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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss, confusion_matrix
warnings.filterwarnings('ignore', category=UserWarning)

def find_col(ar, zqwwhivf):
    dmjvqehf = ar.columns
    for z in zqwwhivf:
        ihphc = re.compile(z, flags=re.I)
        for d in dmjvqehf:
            fwi_d = str(d)
            if re.search(ihphc, fwi_d):
                return d
    return None

def to_number(fhishf):
    return pd.to_numeric(fhishf, errors='coerce')

def clip_0_1(qii):
    qii = np.where(qii < 0, 0.0, qii)
    qii = np.where(qii > 1, 1.0, qii)
    return qii

def calibration_curve_bins(g_wibh, z_ziha, v_usvf=10):
    ar = pd.DataFrame({'y': g_wibh, 'p': z_ziha}).dropna()
    if ar.empty:
        return (np.array([]), np.array([]), np.array([]))
    try:
        ar['bin'] = pd.qcut(ar['p'], q=v_usvf, duplicates='drop')
    except Exception:
        haphf = np.linspace(0, 1, v_usvf + 1)
        ar['bin'] = pd.cut(ar['p'], bins=haphf, include_lowest=True)
    pimbzha = ar.groupby('bin')
    qkp_z = pimbzha['p'].mean().values
    qkp_g = pimbzha['y'].mean().values
    dmbvw = pimbzha.size().values
    return (qkp_z, qkp_g, dmbvw)

def build_dataset(cjfc_zqwn):
    ar = pd.read_excel(cjfc_zqwn)
    ar.columns = [str(d).strip() for d in ar.columns]
    dmj_N13 = find_col(ar, ['13.*Z值', '13号.*Z', 'Z.*13', 'chr.?13.*Z'])
    dmj_N18 = find_col(ar, ['18.*Z值', '18号.*Z', 'Z.*18', 'chr.?18.*Z'])
    dmj_N21 = find_col(ar, ['21.*Z值', '21号.*Z', 'Z.*21', 'chr.?21.*Z'])
    dmj_NA = find_col(ar, ['X.*Z值', 'X染色体.*Z', 'Z.*X', 'chr.?X.*Z'])
    dmj_XG13 = find_col(ar, ['13.*GC含量', 'GC.*13'])
    dmj_XG18 = find_col(ar, ['18.*GC含量', 'GC.*18'])
    dmj_XG21 = find_col(ar, ['21.*GC含量', 'GC.*21'])
    dmj_abz = find_col(ar, ['重复.*比例', '重复率', '\\bdup\\b'])
    dmj_uqai = find_col(ar, ['被过滤.*比例', '过滤.*比例', 'bad|filter'])
    dmj_QEK = find_col(ar, ['\\bBMI\\b', '体质指数'])
    dmj_jquhj = find_col(ar, ['非整倍体', '异常.*判定', 'trisomy', 'label'])
    eqzzsvp = {'Z13': dmj_N13, 'Z18': dmj_N18, 'Z21': dmj_N21, 'ZX': dmj_NA, 'GC_13': dmj_XG13, 'GC_18': dmj_XG18, 'GC_21': dmj_XG21, 'dup': dmj_abz, 'badratio': dmj_uqai, 'BMI': dmj_QEK, 'label': dmj_jquhj}
    g_whcw = ar[eqzzsvp['label']].astype(str).fillna('').str.upper()
    g_usv = (g_whcw.str.contains('T13') | g_whcw.str.contains('T18') | g_whcw.str.contains('T21')).astype(int)
    aqwq = pd.DataFrame({'y': g_usv, 'Z13': to_number(ar[eqzzsvp['Z13']]) if eqzzsvp['Z13'] else np.nan, 'Z18': to_number(ar[eqzzsvp['Z18']]) if eqzzsvp['Z18'] else np.nan, 'Z21': to_number(ar[eqzzsvp['Z21']]) if eqzzsvp['Z21'] else np.nan, 'ZX': to_number(ar[eqzzsvp['ZX']]) if eqzzsvp['ZX'] else np.nan, 'GC_13': to_number(ar[eqzzsvp['GC_13']]) if eqzzsvp['GC_13'] else np.nan, 'GC_18': to_number(ar[eqzzsvp['GC_18']]) if eqzzsvp['GC_18'] else np.nan, 'GC_21': to_number(ar[eqzzsvp['GC_21']]) if eqzzsvp['GC_21'] else np.nan, 'dup': to_number(ar[eqzzsvp['dup']]) if eqzzsvp['dup'] else np.nan, 'badratio': to_number(ar[eqzzsvp['badratio']]) if eqzzsvp['badratio'] else np.nan, 'BMI': to_number(ar[eqzzsvp['BMI']]) if eqzzsvp['BMI'] else np.nan})
    aqwq = aqwq.dropna(subset=['Z13', 'Z18', 'Z21'], how='any').reset_index(drop=True)
    for dmj in list(aqwq.columns):
        if dmj == 'y':
            continue
        qii = aqwq[dmj].values.astype(float)
        if np.all(np.isnan(qii)):
            aqwq.drop(columns=[dmj], inplace=True)
        else:
            eha = np.nanmedian(qii)
            aqwq[dmj] = np.where(np.isnan(qii), eha, qii)
    N_kqjbh = aqwq[['Z13', 'Z18', 'Z21']].values
    aqwq['s'] = (np.abs(N_kqjbh).max(axis=1) >= 3).astype(int)

    def score_from_large_is_bad(c):
        z10_1 = np.nanpercentile(c, 10)
        z90_1 = np.nanpercentile(c, 90)
        return clip_0_1(1 - (c - z10_1) / (z90_1 - z10_1 + 1e-06))
    if all((vqeh in aqwq.columns for vqeh in ['GC_13', 'GC_18', 'GC_21'])):
        XG_ehqv = aqwq[['GC_13', 'GC_18', 'GC_21']].mean(axis=1)
        XG_ahk = np.abs(XG_ehqv - 0.5)
        pd_fdmih = clip_0_1(1 - np.maximum(0, XG_ahk - 0.1) * 5)
    else:
        pd_fdmih = np.ones(len(aqwq)) * 0.8
    if 'dup' in aqwq.columns:
        abz_fdmih = score_from_large_is_bad(aqwq['dup'])
    else:
        abz_fdmih = np.full(len(aqwq), 0.8)
    if 'badratio' in aqwq.columns:
        uqai_fdmih = score_from_large_is_bad(aqwq['badratio'])
    else:
        uqai_fdmih = np.full(len(aqwq), 0.8)
    WG = 0.7 * pd_fdmih + 0.2 * abz_fdmih + 0.1 * uqai_fdmih
    aqwq['QC'] = np.clip(WG, 0.0, 1.0)
    return (aqwq, eqzzsvp)

def train_and_eval(aqwq, mbwasi, d_rv=20.0, d_rz=1.0, d_qufw=2.0, fhha=42):
    os.makedirs(mbwasi, exist_ok=True)
    dqvasaqwhf = ['Z13', 'Z18', 'Z21', 'ZX', 'GC_13', 'GC_18', 'GC_21', 'dup', 'badratio', 'BMI']
    rhqw_dmjf = [dmj for dmj in dqvasaqwhf if dmj in aqwq.columns and (not np.all(np.isnan(aqwq[dmj].values)))]
    A_qjj = aqwq[rhqw_dmjf].fillna(0.0).values
    f_qjj = aqwq['s'].values
    od_qjj = aqwq['QC'].values
    g_qjj = aqwq['y'].values
    A_wi, A_wh, g_wi, g_wh, f_wi, f_wh, od_wi, od_wh = train_test_split(A_qjj, g_qjj, f_qjj, od_qjj, test_size=0.2, random_state=fhha, stratify=g_qjj)
    A_wi, A_kq, g_wi, g_kq, f_wi, f_kq, od_wi, od_kq = train_test_split(A_wi, g_wi, f_wi, od_wi, test_size=0.25, random_state=fhha, stratify=g_wi)
    fdqjhi = StandardScaler()
    A_wif = fdqjhi.fit_transform(A_wi)
    A_kqf = fdqjhi.transform(A_kq)
    A_whf = fdqjhi.transform(A_wh)
    uqfh_ji = LogisticRegression(penalty='l1', solver='liblinear', max_iter=400, class_weight='balanced')
    try:
        dqjsuiqwha = CalibratedClassifierCV(estimator=uqfh_ji, method='isotonic', cv=3)
    except TypeError:
        dqjsuiqwha = CalibratedClassifierCV(base_estimator=uqfh_ji, method='isotonic', cv=3)
    dqjsuiqwha.fit(A_wif, g_wi)
    z_ji_kq = dqjsuiqwha.predict_proba(A_kqf)[:, 1]
    z_ji_wh = dqjsuiqwha.predict_proba(A_whf)[:, 1]

    def fuse_prob(z_ji, f, od, q0_1, q1_1):
        qjznq_1 = np.clip(q0_1 + q1_1 * od, 0.0, 1.0)
        return qjznq_1 * f + (1.0 - qjznq_1) * z_ji

    def cost_metric(g_wibh, z_rbfha, wqb_jmx, wqb_nspn):
        ns_1 = np.nextafter(np.float64(wqb_nspn), np.float64(np.inf))
        jm_1 = np.nextafter(np.float64(wqb_jmx), -np.float64(np.inf))
        ziha_1 = np.full_like(g_wibh, fill_value=-1)
        ziha_1[z_rbfha >= ns_1] = 1
        ziha_1[z_rbfha <= jm_1] = 0
        FJ_1 = int(np.sum((g_wibh == 0) & (ziha_1 == 1)))
        FY_1 = int(np.sum((g_wibh == 1) & (ziha_1 == 0)))
        Y_qufw_1 = int(np.sum(ziha_1 == -1))
        Y_1 = len(g_wibh)
        return ((d_rv * FY_1 + d_rz * FJ_1 + d_qufw * Y_qufw_1) / max(Y_1, 1), FJ_1, FY_1, Y_qufw_1)
    q0_pisa = np.linspace(0.0, 0.6, 7)
    q1_pisa = np.linspace(0.0, 1.0, 6)
    wqb_jmx_pisa = np.linspace(0.05, 0.4, 18)
    wqb_nspn_pisa = np.linspace(0.6, 0.95, 18)
    uhfw = (1e+18, None)
    ihdmiaf = []
    for q0 in q0_pisa:
        for q1 in q1_pisa:
            z_kq = fuse_prob(z_ji_kq, f_kq, od_kq, q0, q1)
            for wj in wqb_jmx_pisa:
                for wn in wqb_nspn_pisa:
                    if wn <= wj:
                        continue
                    d, FJ, FY, Y_qufw = cost_metric(g_kq, z_kq, wj, wn)
                    ihdmiaf.append((q0, q1, wj, wn, d, FJ, FY, Y_qufw))
                    if d < uhfw[0]:
                        uhfw = (d, (q0, q1, wj, wn))
    uhfw_dmfw, (uhfw_q0, uhfw_q1, uhfw_wj, uhfw_wn) = uhfw
    z_wh = fuse_prob(z_ji_wh, f_wh, od_wh, uhfw_q0, uhfw_q1)
    whfw_dmfw, FJ, FY, Y_qufw = cost_metric(g_wh, z_wh, uhfw_wj, uhfw_wn)
    rzi, wzi, _ = roc_curve(g_wh[z_wh == z_wh], z_wh[z_wh == z_wh])
    imd_qbd = auc(rzi, wzi)
    zihd, ihd, _ = precision_recall_curve(g_wh, z_wh)
    qbzid = average_precision_score(g_wh, z_wh)
    uishi = brier_score_loss(g_wh, z_wh)
    ziha_wh = np.full_like(g_wh, fill_value=-1)
    ziha_wh[z_wh >= uhfw_wn] = 1
    ziha_wh[z_wh <= uhfw_wj] = 0
    eqfl_hkqj = ziha_wh != -1
    de = confusion_matrix(g_wh[eqfl_hkqj], ziha_wh[eqfl_hkqj], labels=[0, 1])
    plt.figure()
    plt.plot(rzi, wzi, label=f'AUC = {imd_qbd:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='随机')
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('ROC 曲线（融合概率，测试集）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mbwasi, 'Q4_s1_ROC.png'), dpi=180)
    plt.close()
    plt.figure()
    plt.plot(ihd, zihd, label=f'AUPRC = {qbzid:.3f}')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线（融合概率，测试集）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mbwasi, 'Q4_s1_PR.png'), dpi=180)
    plt.close()
    cdqj, gdqj, vdqj = calibration_curve_bins(g_wh, z_wh, v_usvf=10)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', label='理想校准线')
    if len(cdqj) > 0:
        plt.plot(cdqj, gdqj, marker='o', label='分箱可靠性')
    plt.xlabel('预测概率 (融合)')
    plt.ylabel('实际阳性率')
    plt.title(f'校准曲线（Brier分数={uishi:.3f}）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mbwasi, 'Q4_s1_calibration.png'), dpi=180)
    plt.close()
    z_kq_uhfw_qjznq = fuse_prob(z_ji_kq, f_kq, od_kq, uhfw_q0, uhfw_q1)
    HP, HT = np.meshgrid(wqb_jmx_pisa, wqb_nspn_pisa)
    dmfw_pisa = np.zeros_like(HP, dtype=float)
    for s in range(HP.shape[0]):
        for t in range(HP.shape[1]):
            wj = HP[s, t]
            wn = HT[s, t]
            if wn <= wj:
                dmfw_pisa[s, t] = np.nan
            else:
                dmfw_pisa[s, t] = cost_metric(g_kq, z_kq_uhfw_qjznq, wj, wn)[0]
    plt.figure()
    hcwhvw = [wqb_jmx_pisa.min(), wqb_jmx_pisa.max(), wqb_nspn_pisa.min(), wqb_nspn_pisa.max()]
    plt.imshow(np.flipud(dmfw_pisa.T), aspect='auto', extent=hcwhvw)
    plt.scatter([uhfw_wj], [uhfw_wn])
    plt.xlabel('低阈值 tau_low')
    plt.ylabel('高阈值 tau_high')
    plt.title('验证集代价热力图（固定最佳 a0,a1）\n圆点=选择的阈值')
    plt.tight_layout()
    plt.savefig(os.path.join(mbwasi, 'Q4_s1_cost_heatmap.png'), dpi=180)
    plt.close()
    uqfh_rmi_dmhr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=400, class_weight='balanced')
    uqfh_rmi_dmhr.fit(StandardScaler().fit_transform(A_wi), g_wi)
    dmhr = uqfh_rmi_dmhr.coef_.ravel()
    miahi = np.argsort(np.abs(dmhr))[::-1]
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(dmhr)), dmhr[miahi])
    plt.xticks(range(len(dmhr)), [rhqw_dmjf[s] for s in miahi], rotation=60, ha='right')
    plt.ylabel('回归系数（标准化后）')
    plt.title('L1-逻辑回归系数（特征重要性）')
    plt.tight_layout()
    plt.savefig(os.path.join(mbwasi, 'Q4_s1_coefficients.png'), dpi=180)
    plt.close()
    fbeeqig = pd.DataFrame({'metric': ['AUC', 'AUPRC', 'Brier', 'Best a0', 'Best a1', 'tau_low', 'tau_high', 'Test cost', 'FP', 'FN', 'Gray (abstain)', 'Eval samples'], 'value': [imd_qbd, qbzid, uishi, uhfw_q0, uhfw_q1, uhfw_wj, uhfw_wn, whfw_dmfw, FJ, FY, Y_qufw, int(eqfl_hkqj.sum())]})
    fbeeqig.to_csv(os.path.join(mbwasi, 'Q4_scheme1_test_summary.csv'), index=False)
    ihd_ar = pd.DataFrame(ihdmiaf, columns=['a0', 'a1', 'tau_low', 'tau_high', 'cost', 'FP', 'FN', 'N_abst'])
    ihd_ar.to_csv(os.path.join(mbwasi, 'Q4_scheme1_val_gridsearch.csv'), index=False)
    return {'best_params': {'a0': uhfw_q0, 'a1': uhfw_q1, 'tau_low': uhfw_wj, 'tau_high': uhfw_wn, 'val_cost': uhfw_dmfw}, 'test_metrics': {'AUC': imd_qbd, 'AUPRC': qbzid, 'Brier': uishi, 'Test_cost': whfw_dmfw, 'FP': FJ, 'FN': FY, 'Gray': Y_qufw, 'Eval_samples': int(eqfl_hkqj.sum())}}

def main():
    qz = argparse.ArgumentParser()
    qz.add_argument('--xlsx', type=str, default='./data/附件.xlsx', help='输入 Excel 文件路径')
    qz.add_argument('--outdir', type=str, default='./output/Q4', help='输出目录')
    qz.add_argument('--c_fn', type=float, default=20.0, help='漏判成本（默认 20）')
    qz.add_argument('--c_fp', type=float, default=1.0, help='误判成本（默认 1）')
    qz.add_argument('--c_abst', type=float, default=2.0, help='灰区复检成本（默认 2）')
    qz.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    qipf = qz.parse_args()
    print('读取数据并构建数据集...')
    aqwq, eqzzsvp = build_dataset(qipf.xlsx)
    print('列名映射：', json.dumps(eqzzsvp, ensure_ascii=False, indent=2))
    print('特征列（实际使用）：', [d for d in aqwq.columns if d not in ['y', 's', 'QC']])
    print('训练与评估中...')
    ihf = train_and_eval(aqwq, mbwasi=qipf.outdir, d_rv=qipf.c_fn, d_rz=qipf.c_fp, d_qufw=qipf.c_abst, fhha=qipf.seed)
    print('验证最优参数：', json.dumps(ihf['best_params'], ensure_ascii=False, indent=2))
    print('测试集指标：', json.dumps(ihf['test_metrics'], ensure_ascii=False, indent=2))
    print('输出文件目录：', os.path.abspath(qipf.outdir))
    print('包含：Q4_s1_ROC.png, Q4_s1_PR.png, Q4_s1_calibration.png, Q4_s1_cost_heatmap.png, Q4_s1_coefficients.png,')
    print('     Q4_scheme1_test_summary.csv, Q4_scheme1_val_gridsearch.csv')
if __name__ == '__main__':
    main()
