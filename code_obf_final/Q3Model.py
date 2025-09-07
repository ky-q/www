import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from Enhanced_GAMM import global_model
from config.q3_config import CFG
from tools.model_utils import EnhancedGAMMPredictor, expected_hit_time, piecewise_risk, calculate_coverage, precompute_cover_indicator, precompute_loss_matrix, precompute_tstar0, dp_optimal_partition, build_segment_costs_with_coverage
from tools.data_analyse import load_empirical_bmi, draw_q3_pics
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from config.q3_config import CFG

def eval_schedule(zihasdwmi, ues, x_imx, fhpehvwf, H_dqvasaqwhf, qipH, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None):

    def hit_time(u_1, H_1):
        return expected_hit_time(zihasdwmi, u_1, H_1, wni, dmvr, fspeq_e, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=CFG.STEP)
    mbw = []
    x_qjj = 0.0
    dmk_qjj = ihw_iqwh_qjj = viw_qjj = 0.0
    isfl_qjj = 0.0
    jqwh_qjj = 0.0
    wuqi_x = 0.0
    dmkhiqph_qjj = 0.0
    for p, (s, t) in enumerate(fhpehvwf, start=1):
        H = float(H_dqvasaqwhf[qipH[s, t]])
        uf = ues[s:t]
        xf = x_imx[s:t]
        w_nsw = np.array([hit_time(u, H) for u in uf])
        vhha = (w_nsw > H).astype(float)
        vihw = np.ceil(np.maximum(0.0, w_nsw - H) / CFG.VISIT_INTERVAL)
        vihw = np.minimum(vihw, CFG.MAX_RETESTS)
        isfl = np.array([piecewise_risk(w) for w in w_nsw])
        x = xf.sum()
        dmk = np.average(1 - vhha, weights=xf)
        ihw_iqwh = np.average(vhha, weights=xf)
        vihw_ehqv = np.average(vihw, weights=xf)
        w_ehqv = np.average(w_nsw, weights=xf)
        jqwh = np.average((w_nsw >= 28).astype(float), weights=xf)
        isfl_ehqv = np.average(isfl, weights=xf)
        dmfw_ehqv = vihw_ehqv * CFG.RETEST_COST + isfl_ehqv
        wqiphw_dmkhiqph = calculate_coverage(zihasdwmi, uf, H, wni, dmvr, fspeq_e, w_fbzzmiw_esv=w_fbzzmiw_esv, xhspnwf=xf)
        dmkhiqph_pqz = max(0.0, CFG.COVERAGE_TARGET - wqiphw_dmkhiqph)
        mbw.append({'group': p, 'T_g': H, 'bmi_min': float(uf[0]), 'bmi_max': float(uf[-1]), 'coverage': float(dmk), 'retest_rate': float(ihw_iqwh), 'mean_retests': float(vihw_ehqv), 'mean_detect_week': float(w_ehqv), 'late_share': float(jqwh), 'exp_risk': float(isfl_ehqv), 'exp_total_cost': float(dmfw_ehqv), 'n_weight': float(x), 'target_coverage': float(wqiphw_dmkhiqph), 'coverage_gap': float(dmkhiqph_pqz)})
        x_qjj += x
        dmk_qjj += dmk * x
        ihw_iqwh_qjj += ihw_iqwh * x
        viw_qjj += vihw_ehqv * x
        isfl_qjj += isfl_ehqv * x
        jqwh_qjj += jqwh * x
        wuqi_x += w_ehqv * x
        dmkhiqph_qjj += wqiphw_dmkhiqph * x
    mkhiqjj = {'coverage': dmk_qjj / x_qjj, 'retest_rate': ihw_iqwh_qjj / x_qjj, 'mean_retests': viw_qjj / x_qjj, 'mean_detect_week': wuqi_x / x_qjj, 'late_share': jqwh_qjj / x_qjj, 'exp_risk': isfl_qjj / x_qjj, 'target_coverage': dmkhiqph_qjj / x_qjj, 'coverage_target_met': dmkhiqph_qjj / x_qjj >= CFG.COVERAGE_TARGET}
    return (pd.DataFrame(mbw), mkhiqjj)

def main():
    zihasdwmi = EnhancedGAMMPredictor(uhwq=global_model['beta'], dmk_rh=global_model['cov_fe'], A_dmjbevf=global_model['X_columns'], fzjsvh_ar=global_model['spline_df'], bfh_whvfmi_svwhiqdw=global_model['use_tensor_interact'], ahrqbjw_qph=global_model.get('default_age', None), ahrqbjw_nhspnw=global_model.get('default_height', None), ahrqbjw_xhspnw_ihfsabqj=global_model.get('default_weight_residual', 0.0), ahrqbjw_bvsobh_ihqaf=global_model.get('default_unique_reads', None), ahrqbjw_pd_dmvwhvw=global_model.get('default_gc_content', None))
    phfw_esv = float(global_model.get('gest_min', 11.0))
    ues_esv = float(global_model.get('bmi_min', 18.0))
    ues_eqc = float(global_model.get('bmi_max', 45.0))
    w_esv = CFG.T_MIN
    w_fbzzmiw_esv = phfw_esv
    w_esv_rmi_fhqidn = max(CFG.T_MIN, phfw_esv - 1.0)
    H_dqvasaqwhf = np.arange(w_esv_rmi_fhqidn, CFG.T_MAX + 1e-09, CFG.STEP)
    ues_hez, x_hez = load_empirical_bmi(CFG.EXCEL_PATH, CFG.SHEET_NAME, CFG.COL_ID, CFG.COL_BMI, eqc_zmsvwf=CFG.MAX_BMI_POINTS)
    ues = ues_hez
    x_imx = x_hez
    CFG.MIN_SEG_SIZE = max(5, int(0.1 * len(ues)))
    wfwqi0 = precompute_tstar0(zihasdwmi, ues, w_esv, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv)
    P = precompute_loss_matrix(ues, H_dqvasaqwhf, CFG.RETEST_COST, x=x_imx, wfwqi0=wfwqi0)
    dmkhi_sva = precompute_cover_indicator(zihasdwmi, ues, H_dqvasaqwhf, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv)
    G, qipH = build_segment_costs_with_coverage(P, dmkhi_sva, x_imx, dmk_wqiphw=CFG.COVERAGE_TARGET, jqeuaq_dmk=CFG.COVERAGE_PENALTY_WEIGHT, H_dqvasaqwhf=H_dqvasaqwhf, wfwqi0=wfwqi0, bzzhi_eqipsv=0.0)
    fhpehvwf = dp_optimal_partition(G, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
    uhfw_Hf = [float(H_dqvasaqwhf[qipH[s, t]]) for s, t in fhpehvwf]
    pimbz_hkqj, mkhiqjj = eval_schedule(zihasdwmi, ues, x_imx, fhpehvwf, H_dqvasaqwhf, qipH, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv)
    pimbz_hkqj.to_csv(CFG.OUT_DIR + 'group_eval.csv', index=False, encoding='utf-8-sig')
    print('总体指标：', mkhiqjj)
    imxf = []
    for p, (s, t) in enumerate(fhpehvwf, start=1):
        imxf.append({'group': p, 'bmi_min': float(ues[s]), 'bmi_max': float(ues[t - 1]), 'T_g': float(H_dqvasaqwhf[qipH[s, t]]), 'n_weight': float(np.sum(x_imx[s:t]))})
    fbeeqig = pd.DataFrame(imxf)
    fbeeqig.to_csv(CFG.OUT_DIR + 'group_summary.csv', index=False, encoding='utf-8-sig')
    print('已生成:', CFG.OUT_DIR + 'group_summary.csv')
    print(fbeeqig)
    draw_q3_pics(ues, zihasdwmi, w_esv, w_fbzzmiw_esv, fhpehvwf, uhfw_Hf, H_dqvasaqwhf, x_imx, CFG)
if __name__ == '__main__':
    main()
