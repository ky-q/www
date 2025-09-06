"""
Q3Model - 增强版最优分段模型
- 基于第一问GAMM模型进行BMI分段
- 新增达标比例约束/惩罚
- 使用动态规划求解全局最优分段
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from Enhanced_GAMM import global_model  # 增强版 GAMM 模型
from config.q3_config import CFG
from tools.model_utils import EnhancedGAMMPredictor, expected_hit_time, piecewise_risk, calculate_coverage, precompute_cover_indicator, precompute_loss_matrix, precompute_tstar0, dp_optimal_partition, build_segment_costs_with_coverage
from tools.data_analyse import load_empirical_bmi, draw_q3_pics

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from config.q3_config import CFG

def eval_schedule(predictor, bmi, w_row, segments, T_candidates, argT,
                  thr, conf, sigma_m, t_support_min=None):
    def hit_time(b, T):
        return expected_hit_time(predictor, b, T, thr, conf, sigma_m, t_support_min=t_support_min, step=CFG.STEP)

    out = []
    w_all = 0.0; cov_all = ret_rate_all = nrt_all = 0.0
    risk_all = 0.0; late_all = 0.0; tbar_w = 0.0
    coverage_all = 0.0  # 达标比例（第三问新增）

    for g, (i, j) in enumerate(segments, start=1):
        T = float(T_candidates[argT[i, j]])
        bs = bmi[i:j]; ws = w_row[i:j]
        t_hit = np.array([hit_time(b, T) for b in bs])
        need = (t_hit > T).astype(float)
        nret = np.ceil(np.maximum(0.0, t_hit - T) / CFG.VISIT_INTERVAL)
        nret = np.minimum(nret, CFG.MAX_RETESTS)
        risk = np.array([piecewise_risk(t) for t in t_hit])

        w = ws.sum()
        cov = np.average(1 - need, weights=ws)                  # 首检覆盖率
        ret_rate = np.average(need, weights=ws)                 # 复检率
        nret_mean = np.average(nret, weights=ws)                # 人均复检次数
        t_mean = np.average(t_hit, weights=ws)                  # 平均检出周
        late = np.average((t_hit >= 28).astype(float), weights=ws)  # 晚期占比
        risk_mean = np.average(risk, weights=ws)                # 期望风险
        cost_mean = nret_mean * CFG.RETEST_COST + risk_mean     # 总成本（可选）
        
        # 第三问新增：达标比例评估
        target_coverage = calculate_coverage(
            predictor, bs, T, thr, conf, sigma_m,
            t_support_min=t_support_min, weights=ws
        )
        coverage_gap = max(0.0, CFG.COVERAGE_TARGET - target_coverage)  # 距离目标达标比例的差距

        out.append({
            "group": g, "T_g": T, "bmi_min": float(bs[0]), "bmi_max": float(bs[-1]),
            "coverage": float(cov), "retest_rate": float(ret_rate),
            "mean_retests": float(nret_mean), "mean_detect_week": float(t_mean),
            "late_share": float(late), "exp_risk": float(risk_mean),
            "exp_total_cost": float(cost_mean), "n_weight": float(w),
            # 第三问新增指标
            "target_coverage": float(target_coverage),  # 目标达标比例
            "coverage_gap": float(coverage_gap),  # 达标比例差距
        })

        # overall (weighted by ws)
        w_all += w
        cov_all += cov * w; ret_rate_all += ret_rate * w; nrt_all += nret_mean * w
        risk_all += risk_mean * w; late_all += late * w; tbar_w += t_mean * w
        coverage_all += target_coverage * w  # 第三问：加权平均达标比例

    overall = {
        "coverage": cov_all / w_all,
        "retest_rate": ret_rate_all / w_all,
        "mean_retests": nrt_all / w_all,
        "mean_detect_week": tbar_w / w_all,
        "late_share": late_all / w_all,
        "exp_risk": risk_all / w_all,
        "target_coverage": coverage_all / w_all,  # 第三问：总体达标比例
        "coverage_target_met": (coverage_all / w_all) >= CFG.COVERAGE_TARGET  # 是否满足目标
    }
    return pd.DataFrame(out), overall

def main():
    """主函数"""
    # 1) 预测器 & 支撑范围
    predictor = EnhancedGAMMPredictor(
        beta=global_model["beta"],
        cov_fe=global_model["cov_fe"],
        X_columns=global_model["X_columns"],
        spline_df=global_model["spline_df"],
        use_tensor_interact=global_model["use_tensor_interact"],
        # 第三问新增：扩展协变量
        default_age=global_model.get("default_age", None),
        default_height=global_model.get("default_height", None),
        default_weight_residual=global_model.get("default_weight_residual", 0.0),
        default_unique_reads=global_model.get("default_unique_reads", None),
        default_gc_content=global_model.get("default_gc_content", None),
    )

    gest_min = float(global_model.get("gest_min", 11.0))
    bmi_min = float(global_model.get("bmi_min", 18.0))
    bmi_max = float(global_model.get("bmi_max", 45.0))

    # 下限策略
    t_min = CFG.T_MIN
    t_support_min = gest_min

    # 候选统一时点 T
    t_min_for_search = max(CFG.T_MIN, gest_min - 1.0)
    T_candidates = np.arange(t_min_for_search, CFG.T_MAX + 1e-9, CFG.STEP)

    # 2) BMI：使用 Excel 的真实分布（按孕妇去重；可压缩并带权重）
    bmi_emp, w_emp = load_empirical_bmi(
        CFG.EXCEL_PATH, CFG.SHEET_NAME, CFG.COL_ID, CFG.COL_BMI, max_points=CFG.MAX_BMI_POINTS
    )
    bmi = bmi_emp
    w_row = w_emp

    # 最小段长按“代表点数量”设定
    CFG.MIN_SEG_SIZE = max(5, int(0.10 * len(bmi)))

    # 3) 段成本矩阵 & DP（固定段数）
    tstar0 = precompute_tstar0(predictor, bmi, t_min, CFG.THRESHOLD, CFG.CONF_LEVEL,
                           CFG.SIGMA_M, t_support_min=t_support_min)

    L = precompute_loss_matrix(bmi, T_candidates, CFG.RETEST_COST, w=w_row, tstar0=tstar0)

    cover_ind = precompute_cover_indicator(
        predictor, bmi, T_candidates,
        CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M,
        t_support_min=t_support_min
    )
    C, argT = build_segment_costs_with_coverage(
        L, cover_ind, w_row,
        cov_target=CFG.COVERAGE_TARGET,
        lambda_cov=CFG.COVERAGE_PENALTY_WEIGHT,
        T_candidates=T_candidates,
        tstar0=tstar0,
        upper_margin=0.0   # 严格不超过段内 max t*
    )

    segments = dp_optimal_partition(C, CFG.N_GROUPS, CFG.MIN_SEG_SIZE)
    best_Ts = [float(T_candidates[argT[i, j]]) for (i, j) in segments]

    group_eval, overall = eval_schedule(
        predictor, bmi, w_row, segments, T_candidates, argT,
        CFG.THRESHOLD, CFG.CONF_LEVEL, CFG.SIGMA_M, t_support_min=t_support_min
    )
    group_eval.to_csv(CFG.OUT_DIR + "group_eval.csv", index=False, encoding="utf-8-sig")
    print("总体指标：", overall)


    # 4) 导出分组摘要
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
    summary.to_csv(CFG.OUT_DIR + "group_summary.csv", index=False, encoding="utf-8-sig")
    print("已生成:", CFG.OUT_DIR + "group_summary.csv")
    print(summary)

    draw_q3_pics(bmi, predictor, t_min, t_support_min, segments, best_Ts, T_candidates, w_row, CFG)

    
if __name__ == "__main__":
    main()