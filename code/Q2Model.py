import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from GAMM import global_model
from tools.data_process import piecewise_risk
from tools.data_analyse import eval_schedule, load_empirical_bmi, draw_q2_pics
from tools.model_utils import (
    GAMMPredictor, expected_hit_time,
    precompute_loss_matrix, dp_optimal_partition, precompute_tstar0,
    build_segment_costs_simple
)
from config.constant import CFG
def individual_loss(pred, b, T, thr, conf, sigma_m, cost, t_support_min=None):
    t_hit = expected_hit_time(pred, b, T, thr, conf, sigma_m, t_support_min=t_support_min)
    gap = max(0.0, t_hit - T)
    n_retests = int(np.ceil(gap / CFG.VISIT_INTERVAL))
    n_retests = min(n_retests, CFG.MAX_RETESTS)
    retest_cost = n_retests * cost
    risk = piecewise_risk(t_hit)
    return CFG.FIRST_VISIT_COST + retest_cost + risk
def main():
    predictor = GAMMPredictor(
        beta=global_model["beta"],
        cov_fe=global_model["cov_fe"],
        X_columns=global_model["X_columns"],
        spline_df=global_model["spline_df"],
        use_tensor_interact=global_model["use_tensor_interact"],
    )
    gest_min = float(global_model.get("gest_min", 11.0))
    bmi_min = float(global_model.get("bmi_min", 18.0))
    bmi_max = float(global_model.get("bmi_max", 45.0))
    t_min = CFG.T_MIN_RAW
    t_support_min = gest_min
    t_min_for_search = max(CFG.T_MIN_RAW, gest_min - 1.0)
    T_candidates = np.arange(t_min_for_search, CFG.T_MAX + 1e-9, CFG.STEP)
    bmi_emp, w_emp = load_empirical_bmi(
        CFG.EXCEL_PATH, CFG.SHEET_NAME, CFG.COL_ID, CFG.COL_BMI, max_points=CFG.MAX_BMI_POINTS
    )
    bmi = bmi_emp
    w_row = w_emp
    CFG.MIN_SEG_SIZE = max(5, int(0.10 * len(bmi)))
    tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                           CFG.SIGMA_M, t_support_min=t_support_min)
    L = precompute_loss_matrix(bmi, T_candidates, CFG.RETEST_COST, w=w_row, tstar0=tstar0)
    C, argT = build_segment_costs_simple(L)
    segments = dp_optimal_partition(C, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
    best_Ts = [float(T_candidates[argT[i, j]]) for (i, j) in segments]
    group_eval, overall = eval_schedule(
        predictor, bmi, w_row, segments, T_candidates, argT,
        CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, t_support_min=t_support_min
    )
    group_eval.to_csv(CFG.Q2PicPath + "group_eval.csv", index=False, encoding="utf-8-sig")
    print("总体指标：", overall)
    rows = []
    for g, (i, j) in enumerate(segments, start=1):
        rows.append({
            "group": g,
            "bmi_min": float(bmi[i]),
            "bmi_max": float(bmi[j - 1]),
            "T_g": float(T_candidates[argT[i, j]]),
            "n_weight": float(np.sum(w_row[i:j]))
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(CFG.Q2PicPath + CFG.OUT_GROUP_SUMMARY, index=False, encoding="utf-8-sig")
    print("已生成:",CFG.Q2PicPath + CFG.OUT_GROUP_SUMMARY)
    print(summary)
    draw_q2_pics(bmi, predictor, t_min, t_support_min, segments, best_Ts, T_candidates, w_row)
if __name__ == "__main__":
    main()
