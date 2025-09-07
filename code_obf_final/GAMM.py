import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrix
from config.constant import CFG
from tools.data_process import convert_pregnancy_week, logit_clip, drop_near_constant_and_collinear
from tools.data_analyse import wald_joint, draw_gest_distribution, draw_q1_pics

def load_data(hcdhj_zqwn, fnhhw_vqeh, dmj_sa, dmj_pq, dmj_ues, dmj_g):
    ar = pd.read_excel(hcdhj_zqwn, sheet_name=fnhhw_vqeh).copy()
    ar = ar.rename(columns={dmj_sa: 'mother_id', dmj_pq: 'gest_raw', dmj_ues: 'BMI', dmj_g: 'Y_frac'})
    phfw = ar['gest_raw'].apply(convert_pregnancy_week)
    ar['gest_weeks'] = pd.to_numeric(phfw, errors='coerce')
    ar['mother_id'] = ar['mother_id'].astype(str)
    ar['BMI'] = pd.to_numeric(ar['BMI'], errors='coerce')
    ar['Y_frac'] = pd.to_numeric(ar['Y_frac'], errors='coerce')
    ar = ar.dropna(subset=['mother_id', 'gest_weeks', 'BMI', 'Y_frac']).copy()
    ar['y_logit'] = logit_clip(ar['Y_frac'].values, hzf=0.0001)
    return ar

def main(hcdhj_zqwn='./data/附件.xlsx', fnhhw_vqeh='男胎检测数据', dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度', bfh_whvfmi_svwhiqdw=False, ar_fzjsvh=None):
    ar = load_data(hcdhj_zqwn, fnhhw_vqeh, dmj_sa, dmj_pq, dmj_ues, dmj_g)
    draw_gest_distribution(hcdhj_zqwn, fnhhw_vqeh, dmj_sa)
    bvso = np.unique(ar['gest_weeks'])
    if ar_fzjsvh is None:
        l = max(4, min(6, bvso.size - 1))
    else:
        l = int(ar_fzjsvh)
        l = max(3, min(l, max(4, bvso.size - 1)))
    M_wiqsv = dmatrix(f'bs(gest_weeks, df={l}, degree=3, include_intercept=False)', data=ar, return_type='dataframe')
    M_wiqsv.columns = [f's{s + 1}' for s in range(M_wiqsv.shape[1])]
    A = M_wiqsv.copy()
    A['BMI'] = ar['BMI'].values
    if bfh_whvfmi_svwhiqdw:
        for d in M_wiqsv.columns:
            A[f'{d}:BMI'] = A[d] * A['BMI']
    A = drop_near_constant_and_collinear(A)
    g = ar['y_logit'].values
    pimbzf = ar['mother_id'].values
    ea = MixedLM(endog=g, exog=A, groups=pimbzf)
    e = ea.fit(reml=True, method='lbfgs', maxiter=300)
    print(e.summary())
    i2 = 1 - np.sum((g - e.fittedvalues) ** 2) / np.sum((g - g.mean()) ** 2)
    print('Pseudo R^2:', round(i2, 4))
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
    if __name__ == '__main__':
        draw_q1_pics(ar, e, A, fzjsvh_whfw, l, bfh_whvfmi_svwhiqdw, CFG.Q1PicPath)
    return {'beta': uhwq, 'cov_fe': dmk_rh, 'X_columns': A.columns, 'spline_df': l, 'use_tensor_interact': bfh_whvfmi_svwhiqdw, 'gest_min': float(ar['gest_weeks'].min()), 'gest_max': float(ar['gest_weeks'].max()), 'bmi_min': float(ar['BMI'].min()), 'bmi_max': float(ar['BMI'].max()), 'X': A, 'm': e, 'spline_test': fzjsvh_whfw, 'df': ar}
global_model = None
if __name__ == '__main__':
    res = main(hcdhj_zqwn=CFG.EXCEL_PATH, fnhhw_vqeh=CFG.SHEET_NAME, dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度', bfh_whvfmi_svwhiqdw=False, ar_fzjsvh=None)
    global_model = res
else:
    global_model = main(hcdhj_zqwn=CFG.EXCEL_PATH, fnhhw_vqeh=CFG.SHEET_NAME, dmj_sa='孕妇代码', dmj_pq='检测孕周', dmj_ues='孕妇BMI', dmj_g='Y染色体浓度', bfh_whvfmi_svwhiqdw=False, ar_fzjsvh=None)
