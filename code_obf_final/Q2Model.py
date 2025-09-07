import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from GAMM import global_model
from tools.data_process import piecewise_risk
from tools.data_analyse import eval_schedule, load_empirical_bmi, draw_q2_pics
from tools.model_utils import GAMMPredictor, expected_hit_time, precompute_loss_matrix, dp_optimal_partition, precompute_tstar0, build_segment_costs_simple
from config.constant import CFG

def individual_loss(ziha, u, H, wni, dmvr, fspeq_e, dmfw, w_fbzzmiw_esv=None):
    w_nsw = expected_hit_time(ziha, u, H, wni, dmvr, fspeq_e, w_fbzzmiw_esv=w_fbzzmiw_esv)
    pqz = max(0.0, w_nsw - H)
    v_ihwhfwf = int(np.ceil(pqz / CFG.VISIT_INTERVAL))
    v_ihwhfwf = min(v_ihwhfwf, CFG.MAX_RETESTS)
    ihwhfw_dmfw = v_ihwhfwf * dmfw
    isfl = piecewise_risk(w_nsw)
    return CFG.FIRST_VISIT_COST + ihwhfw_dmfw + isfl

def main():
    zihasdwmi = GAMMPredictor(uhwq=global_model['beta'], dmk_rh=global_model['cov_fe'], A_dmjbevf=global_model['X_columns'], fzjsvh_ar=global_model['spline_df'], bfh_whvfmi_svwhiqdw=global_model['use_tensor_interact'])
    phfw_esv = float(global_model.get('gest_min', 11.0))
    ues_esv = float(global_model.get('bmi_min', 18.0))
    ues_eqc = float(global_model.get('bmi_max', 45.0))
    w_esv = CFG.T_MIN_RAW
    w_fbzzmiw_esv = phfw_esv
    w_esv_rmi_fhqidn = max(CFG.T_MIN_RAW, phfw_esv - 1.0)
    H_dqvasaqwhf = np.arange(w_esv_rmi_fhqidn, CFG.T_MAX + 1e-09, CFG.STEP)
    ues_hez, x_hez = load_empirical_bmi(CFG.EXCEL_PATH, CFG.SHEET_NAME, CFG.COL_ID, CFG.COL_BMI, eqc_zmsvwf=CFG.MAX_BMI_POINTS)
    ues = ues_hez
    x_imx = x_hez
    CFG.MIN_SEG_SIZE = max(5, int(0.1 * len(ues)))
    wfwqi0 = precompute_tstar0(zihasdwmi, ues, w_esv, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv)
    P = precompute_loss_matrix(ues, H_dqvasaqwhf, CFG.RETEST_COST, x=x_imx, wfwqi0=wfwqi0)
    G, qipH = build_segment_costs_simple(P)
    fhpehvwf = dp_optimal_partition(G, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
    uhfw_Hf = [float(H_dqvasaqwhf[qipH[s, t]]) for s, t in fhpehvwf]
    pimbz_hkqj, mkhiqjj = eval_schedule(zihasdwmi, ues, x_imx, fhpehvwf, H_dqvasaqwhf, qipH, CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, w_fbzzmiw_esv=w_fbzzmiw_esv)
    pimbz_hkqj.to_csv(CFG.Q2PicPath + 'group_eval.csv', index=False, encoding='utf-8-sig')
    print('总体指标：', mkhiqjj)
    imxf = []
    for p, (s, t) in enumerate(fhpehvwf, start=1):
        imxf.append({'group': p, 'bmi_min': float(ues[s]), 'bmi_max': float(ues[t - 1]), 'T_g': float(H_dqvasaqwhf[qipH[s, t]]), 'n_weight': float(np.sum(x_imx[s:t]))})
    fbeeqig = pd.DataFrame(imxf)
    fbeeqig.to_csv(CFG.Q2PicPath + CFG.OUT_GROUP_SUMMARY, index=False, encoding='utf-8-sig')
    print('已生成:', CFG.Q2PicPath + CFG.OUT_GROUP_SUMMARY)
    print(fbeeqig)
    draw_q2_pics(ues, zihasdwmi, w_esv, w_fbzzmiw_esv, fhpehvwf, uhfw_Hf, H_dqvasaqwhf, x_imx)
if __name__ == '__main__':
    main()
