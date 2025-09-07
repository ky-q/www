import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint
from config.constant import CFG
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def process_extended_covariates(ar):
    if '年龄' in ar.columns:
        ar.rename(columns={'年龄': 'age'}, inplace=True)
    if '身高' in ar.columns:
        ar.rename(columns={'身高': 'height'}, inplace=True)
    if '体重' in ar.columns:
        ar.rename(columns={'体重': 'weight'}, inplace=True)
    if '唯一比对的读段数' in ar.columns:
        ar.rename(columns={'唯一比对的读段数': 'unique_reads'}, inplace=True)
    if 'GC含量' in ar.columns:
        ar.rename(columns={'GC含量': 'gc_content'}, inplace=True)
    return ar

def load_data(hcdhj_zqwn, fnhhw_vqeh, dmj_sa, dmj_pq, dmj_ues, dmj_g):
    ar = pd.read_excel(hcdhj_zqwn, sheet_name=fnhhw_vqeh).copy()
    ar = ar.rename(columns={dmj_sa: 'mother_id', dmj_pq: 'gest_raw', dmj_ues: 'BMI', dmj_g: 'Y_frac'})
    ar = process_extended_covariates(ar)
    phfw = ar['gest_raw'].apply(convert_pregnancy_week)
    ar['gest_weeks'] = pd.to_numeric(phfw, errors='coerce')
    ar['mother_id'] = ar['mother_id'].astype(str)
    ar['BMI'] = pd.to_numeric(ar['BMI'], errors='coerce')
    ar['Y_frac'] = pd.to_numeric(ar['Y_frac'], errors='coerce')
    if 'age' in ar.columns:
        ar['age'] = pd.to_numeric(ar['age'], errors='coerce')
    else:
        ar['age'] = np.nan
    if 'height' in ar.columns:
        ar['height'] = pd.to_numeric(ar['height'], errors='coerce')
        ar['height_orig'] = ar['height'].copy()
        nhspnw_ehqv = ar['height'].mean()
        nhspnw_fwa = ar['height'].std()
        if nhspnw_fwa > 0:
            ar['height'] = (ar['height'] - nhspnw_ehqv) / nhspnw_fwa
    else:
        ar['height'] = np.nan
        ar['height_orig'] = np.nan
    if 'weight' in ar.columns:
        ar['weight'] = pd.to_numeric(ar['weight'], errors='coerce')
    else:
        ar['weight'] = np.nan
    if 'unique_reads' in ar.columns:
        ar['unique_reads'] = pd.to_numeric(ar['unique_reads'], errors='coerce')
        ar['unique_reads'] = ar['unique_reads'] / 1000000.0
    else:
        ar['unique_reads'] = np.nan
    if 'gc_content' in ar.columns:
        ar['gc_content'] = pd.to_numeric(ar['gc_content'], errors='coerce')
        pd_ehqv = ar['gc_content'].mean()
        pd_fwa = ar['gc_content'].std()
        if pd_fwa > 0:
            ar['gc_content'] = (ar['gc_content'] - pd_ehqv) / pd_fwa
    else:
        ar['gc_content'] = np.nan
    ar = ar.dropna(subset=['mother_id', 'gest_weeks', 'BMI', 'Y_frac']).copy()
    if not ar['weight'].isna().all() and (not ar['height_orig'].isna().all()):
        emahj_xhspnw = np.polyfit(ar['BMI'] * (ar['height_orig'] / 100) ** 2, ar['weight'], 1)
        hczhdwha_xhspnw = emahj_xhspnw[0] * ar['BMI'] * (ar['height_orig'] / 100) ** 2 + emahj_xhspnw[1]
        ar['weight_residual'] = ar['weight'] - hczhdwha_xhspnw
    else:
        ar['weight_residual'] = np.nan
    ar['y_logit'] = logit_clip(ar['Y_frac'].values, hzf=0.0001)
    return ar

def create_design_matrix(ar, l, bfh_whvfmi_svwhiqdw=False):
    M_wiqsv = dmatrix(f'bs(gest_weeks, df={l}, degree=3, include_intercept=False)', data=ar, return_type='dataframe')
    M_wiqsv.columns = [f's{s + 1}' for s in range(M_wiqsv.shape[1])]
    A = M_wiqsv.copy()
    A['BMI'] = ar['BMI'].values
    if bfh_whvfmi_svwhiqdw:
        for d in M_wiqsv.columns:
            A[f'{d}:BMI'] = A[d] * A['BMI']
    if 'age' in ar.columns and (not ar['age'].isna().all()):
        A['age'] = ar['age'].values
    if 'height' in ar.columns and (not ar['height'].isna().all()):
        A['height'] = ar['height'].values
    if 'weight_residual' in ar.columns and (not ar['weight_residual'].isna().all()):
        A['weight_residual'] = ar['weight_residual'].values
    if 'unique_reads' in ar.columns and (not ar['unique_reads'].isna().all()):
        A['unique_reads'] = ar['unique_reads'].values
    if 'gc_content' in ar.columns and (not ar['gc_content'].isna().all()):
        A['gc_content'] = ar['gc_content'].values
    A = drop_near_constant_and_collinear(A)
    return A

def main(hcdhj_zqwn=CFG.EXCEL_PATH, fnhhw_vqeh='男胎检测数据', dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度', bfh_whvfmi_svwhiqdw=False, ar_fzjsvh=None):
    ar = load_data(hcdhj_zqwn, fnhhw_vqeh, dmj_sa, dmj_pq, dmj_ues, dmj_g)
    bvso = np.unique(ar['gest_weeks'])
    if ar_fzjsvh is None:
        l = max(4, min(6, bvso.size - 1))
    else:
        l = int(ar_fzjsvh)
        l = max(3, min(l, max(4, bvso.size - 1)))
    A = create_design_matrix(ar, l, bfh_whvfmi_svwhiqdw)
    g = ar['y_logit'].values
    pimbzf = ar['mother_id'].values
    ea = MixedLM(endog=g, exog=A, groups=pimbzf)
    e = ea.fit(reml=True, method='lbfgs', maxiter=300)
    print(e.summary())
    i2 = 1 - np.sum((g - e.fittedvalues) ** 2) / np.sum((g - g.mean()) ** 2)
    print('Pseudo R^2:', round(i2, 4))
    ues_whfw = None
    if 'BMI' in A.columns:
        print('\n[Wald] H0: BMI = 0')
        ues_ihf = e.wald_test('BMI = 0', use_f=True)
        print(ues_ihf.summary())
    fzjsvh_dmjf = [d for d in A.columns if d.startswith('s')]
    print('\n[Wald] 样条整体 H0: 所有 s(gest_weeks) 系数 = 0')
    fzjsvh_whfw = wald_joint(fzjsvh_dmjf, e, A)
    if bfh_whvfmi_svwhiqdw:
        svwhi_dmjf = [d for d in A.columns if d.endswith(':BMI')]
        print('\n[Wald] 样条×BMI 交互整体 H0: 交互项=0')
        wald_joint(svwhi_dmjf, e, A)
    rh_vqehf = list(A.columns)
    v_rh = len(rh_vqehf)
    zqiqef_qjj = np.asarray(e.params).ravel()
    uhwq = pd.Series(zqiqef_qjj[:v_rh], index=rh_vqehf)
    dmk_qjj = np.asarray(e.cov_params())
    dmk_rh = dmk_qjj[:v_rh, :v_rh]
    rsjwhiha_dmjbevf = [dmj for dmj in A.columns if dmj != 'weight_residual' and dmj != 'age']
    rsjwhiha_uhwq = uhwq.loc[rsjwhiha_dmjbevf]
    if dmk_rh is not None:
        svasdhf = [s for s, dmj in enumerate(A.columns) if dmj in rsjwhiha_dmjbevf]
        rsjwhiha_dmk_rh = dmk_rh[np.ix_(svasdhf, svasdhf)]
    else:
        rsjwhiha_dmk_rh = None
    ihfbjw = {'beta': rsjwhiha_uhwq, 'cov_fe': rsjwhiha_dmk_rh, 'X_columns': rsjwhiha_dmjbevf, 'formula': f'bs(GA, df={l}, include_intercept=True)', 'spline_df': l, 'use_tensor_interact': bfh_whvfmi_svwhiqdw, 'gest_min': float(ar['gest_weeks'].min()), 'gest_max': float(ar['gest_weeks'].max()), 'bmi_min': float(ar['BMI'].min()), 'bmi_max': float(ar['BMI'].max()), 'default_age': None, 'default_height': 0.0, 'default_weight_residual': None, 'default_unique_reads': float(ar['unique_reads'].median()) if not ar['unique_reads'].isna().all() else None, 'default_gc_content': 0.0}
    from tools.data_analyse import draw_q1_pics
    if __name__ == '__main__':
        draw_q1_pics(ar, e, A, fzjsvh_whfw, l, bfh_whvfmi_svwhiqdw, CFG.Q3PicPath)
    return ihfbjw
global_model = None
if __name__ == '__main__':
    global_model = main(hcdhj_zqwn=CFG.EXCEL_PATH, fnhhw_vqeh=CFG.SHEET_NAME, dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度', bfh_whvfmi_svwhiqdw=True, ar_fzjsvh=None)
else:
    global_model = main(hcdhj_zqwn=CFG.EXCEL_PATH, fnhhw_vqeh=CFG.SHEET_NAME, bfh_whvfmi_svwhiqdw=True)
