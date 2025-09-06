"""
Common utility functions for model prediction and evaluation
to break circular imports between RAMM.py, Q2Model.py, and data_analyse.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from config.constant import CFG
from tools.data_process import piecewise_risk

def logistic(x): return 1.0 / (1.0 + np.exp(-x))

def z_from_conf(level): return norm.ppf(level)

def _softplus(x, beta=4.0):
    return (1.0 / beta) * np.log1p(np.exp(beta * x))

def lcl_prob_from_logit(eta, se, conf, sigma_m, t=None, t_support_min=None):
    """置信下界（带可选软下限惩罚）"""
    z = norm.ppf(conf)
    extra_var = 0.0
    if (t is not None) and (t_support_min is not None):
        t = np.asarray(t, dtype=float)
        d = (t_support_min + CFG.SOFT_BAND) - t
        gap = _softplus(d, beta=CFG.SOFT_BETA)
        extra_sigma = CFG.EXTRAP_SIGMA_PER_WEEK * (gap ** CFG.SOFT_POWER)
        extra_var = (extra_sigma ** 2)
    s_eff = np.sqrt(se**2 + sigma_m**2 + extra_var)
    return 1.0 / (1.0 + np.exp(-(eta - z * s_eff)))

def first_hit_time_for_b(pred, b, t_min, t_max, thr, conf, sigma_m, t_support_min=None, step=0.1):
    """找到首次达标时间"""
    ts = np.arange(t_min, t_max + 1e-9, step)
    bb = np.full_like(ts, b, dtype=float)
    eta, se = pred.predict_logit_and_se(ts, bb)
    lcl = lcl_prob_from_logit(eta, se, conf, sigma_m, t=ts, t_support_min=t_support_min)
    ge = np.where(lcl >= thr)[0]
    if ge.size == 0:
        return None
    j = ge[0]
    if j == 0:
        return float(ts[0])
    t0, t1 = ts[j - 1], ts[j]
    y0, y1 = lcl[j - 1], lcl[j]
    w = (thr - y0) / (y1 - y0) if y1 != y0 else 1.0
    return float(t0 + w * (t1 - t0))

def expected_hit_time(pred, b, T, thr, conf, sigma_m, t_support_min=None, step=CFG.STEP):
    """期望达标时间"""
    ts = np.arange(T, CFG.T_MAX + 1e-9, step)
    bb = np.full_like(ts, b, dtype=float)
    eta, se = pred.predict_logit_and_se(ts, bb)
    lcl = lcl_prob_from_logit(eta, se, conf, sigma_m, t=ts, t_support_min=t_support_min)
    ge = np.where(lcl >= thr)[0]
    if ge.size == 0:
        return float(ts[-1])
    return float(ts[ge[0]])

def precompute_tstar0(predictor, bmi, t_min, thr, conf, sigma_m, t_support_min=None):
    """预计算所有BMI的首次达标时间"""
    return np.array([
        first_hit_time_for_b(predictor, float(b), t_min, CFG.T_MAX,
                             thr, conf, sigma_m, t_support_min=t_support_min, step=CFG.STEP)
        for b in bmi
    ], dtype=float)

def precompute_loss_matrix(bmi, Tcand, cost, w=None, tstar0=None):
    """预计算损失矩阵"""
    # 要求传入 tstar0 = precompute_tstar0(...)
    n, m = len(bmi), len(Tcand)
    if tstar0 is None:
        raise ValueError("precompute_loss_matrix 需要 tstar0")
    L = np.zeros((n, m))
    for i, t0 in enumerate(tstar0):
        for j, T in enumerate(Tcand):
            # 复检：当 T < t*(b)
            gap = max(0.0, t0 - T)
            n_retests = int(np.ceil(gap / CFG.VISIT_INTERVAL))
            n_retests = min(n_retests, CFG.MAX_RETESTS)
            retest_cost = n_retests * cost

            # 检出时刻与风险
            t_hit = T if T >= t0 else t0
            risk = piecewise_risk(t_hit)

            # 等待罚：当 T > t*(b)
            wait_pen = CFG.WAIT_PENALTY_PER_WEEK * max(0.0, T - t0)

            L[i, j] = CFG.FIRST_VISIT_COST + retest_cost + risk + wait_pen

    if w is not None:
        L = (L.T * np.asarray(w, float)).T
    return L

def build_segment_costs_simple(L):
    """
    C[i,j] = min_T sum_{u=i+1..j} L[u,T]
    argT[i,j] = 使段成本最小的 T 的索引
    """
    n, m = L.shape
    S = np.zeros((n + 1, m))
    for k in range(1, n + 1):
        S[k] = S[k - 1] + L[k - 1]
    C = np.full((n + 1, n + 1), np.inf)
    argT = np.full((n + 1, n + 1), -1, dtype=int)
    for i in range(n):
        for j in range(i + 1, n + 1):
            seg = S[j] - S[i]          # 该段在所有 T 下的合计损失 (m,)
            t_idx = int(np.argmin(seg))
            C[i, j] = float(seg[t_idx])
            argT[i, j] = t_idx
    return C, argT

def build_segment_costs_with_coverage(L, cover_ind=None, w_row=None,
                        cov_target=0.85,
                        lambda_cov=50.0,
                        *,
                        T_candidates=None,
                        tstar0=None,
                        upper_margin=0.0):
    """
    段成本（带可选覆盖率约束），并保证 T 不超过该段的 max t*。
    参数：
      L:            (n_bmi, n_T) 纯成本矩阵（不含覆盖率惩罚）
      cover_ind:    (n_bmi, n_T) 0/1，1{ p_LCL(T;b) >= THRESHOLD }，仅在 use_constraint=True 时需要
      w_row:        (n_bmi,)     代表点权重
      T_candidates: (n_T,)       候选统一时点的数值（用于上界判定）
      tstar0:       (n_bmi,)     每个 BMI 的最早达标周
      enforce_T_upper:           True=强制 T ≤ 段内 max t* (+upper_margin)
      upper_margin:              上界缓冲，严格就设 0
    """
    if T_candidates is None:
        raise ValueError("build_segment_costs 需要 T_candidates")
    n, m = L.shape

    # 纯成本的前缀和，O(1) 取段合计
    S = np.zeros((n + 1, m), dtype=float)
    for k in range(1, n + 1):
        S[k] = S[k - 1] + L[k - 1]

    if w_row is None:
        w_row = np.ones(n, dtype=float)
    else:
        w_row = np.asarray(w_row, float)

    C = np.full((n + 1, n + 1), np.inf, dtype=float)
    argT = np.full((n + 1, n + 1), -1, dtype=int)

    for i in range(n):
        for j in range(i + 1, n + 1):
            # 该段在所有 T 下的“纯成本合计”
            seg_cost = S[j] - S[i]      # (m,)

            # 覆盖率（段级）硬/软约束
            w_seg = w_row[i:j]
            W = float(w_seg.sum()) if j > i else 1.0
            cov_vec = (cover_ind[i:j] * w_seg[:, None]).sum(axis=0) / W  # (m,)

            penalty = lambda_cov * np.maximum(0.0, cov_target - cov_vec)
            seg_cost = seg_cost + penalty

            # ☆ 上界：不允许 T 超过该段的 max t*
            if tstar0 is not None:
                t_hi = float(np.max(tstar0[i:j]))
                too_late = (np.asarray(T_candidates, float) > (t_hi + upper_margin))
                if np.any(too_late):
                    seg_cost = seg_cost.copy()
                    seg_cost[too_late] = np.inf

            # ☆ 并列选早：给成本加极小偏置，索引越大（越晚）加得越多
            eps = 1e-9
            seg_cost_tiebreak = seg_cost + eps * np.arange(m, dtype=float)
            t_idx = int(np.argmin(seg_cost_tiebreak))

            C[i, j] = float(seg_cost_tiebreak[t_idx])
            argT[i, j] = t_idx

    return C, argT

def dp_optimal_partition(C, K, min_seg):
    """DP：固定段数（全局最优）"""
    n = C.shape[0] - 1
    dp = np.full((K + 1, n + 1), 1e30)
    prev = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k * min_seg, n + 1):
            best, argi = 1e30, -1
            for i in range((k - 1) * min_seg, j - min_seg + 1):
                val = dp[k - 1, i] + C[i, j]
                if val < best:
                    best, argi = val, i
            dp[k, j] = best
            prev[k, j] = argi
    segs = []
    j = n
    for k in range(K, 0, -1):
        i = prev[k, j]
        if i < 0:
            raise RuntimeError("DP 不可行，请检查 MIN_SEG_SIZE。")
        segs.append((i, j))
        j = i
    return segs[::-1]

def precompute_cover_indicator(predictor, bmi, T_candidates, thr, conf, sigma_m, t_support_min=None):
    """
    返回 (n_bmi, n_T) 的 0/1 矩阵：cover_ind[i,j] = 1{ p_LCL(T_j; b_i) >= thr }
    thr 一定要用题面阈值（建议 CFG.THRESHOLD = 0.04）
    """
    n = len(bmi)
    m = len(T_candidates)
    cover = np.zeros((n, m), dtype=float)
    for j, T in enumerate(T_candidates):
        t_vec = np.full(n, float(T), dtype=float)
        eta, se = predictor.predict_logit_and_se(t_vec, bmi)
        p_lcl = lcl_prob_from_logit(eta, se, conf, sigma_m, t=float(T), t_support_min=t_support_min)
        cover[:, j] = (p_lcl >= thr).astype(float)
    return cover

def calculate_coverage(pred, b_values, T, thr, conf, sigma_m,
                       t_support_min=None, weights=None,
                       age=None, height=None, weight_residual=None,
                       unique_reads=None, gc_content=None):
    """
    计算在时点 T 的首检达标比例：
      1{ LCL_Y(T; X) >= thr } 的加权平均
    - thr 应该是 0.04（题面阈值），而不是概率 0.5
    """
    T = float(T)
    b_values = np.asarray(b_values, dtype=float)
    hits = []
    for b in b_values:
        eta, se = pred.predict_logit_and_se(T, b,
                                            age=age, height=height,
                                            weight_residual=weight_residual,
                                            unique_reads=unique_reads,
                                            gc_content=gc_content)
        p = lcl_prob_from_logit(eta[0], se[0], conf, sigma_m,
                                t=T, t_support_min=t_support_min)
        hits.append(1.0 if p >= thr else 0.0)
    hits = np.asarray(hits, float)
    if weights is None:
        return float(hits.mean())
    w = np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(hits * w))
# GAMMPredictor class 
class GAMMPredictor:
    def __init__(self, beta, cov_fe, X_columns, spline_df, use_tensor_interact=False):
        self.beta = np.asarray(beta)
        self.cov_fe = np.asarray(cov_fe) if cov_fe is not None else None
        self.cols = list(X_columns)
        self.k = int(spline_df)
        self.use_tensor_interact = use_tensor_interact

    def predict_logit_and_se(self, t, b):
        from patsy import dmatrix
        import pandas as pd
        df_new = pd.DataFrame({"gest_weeks": t, "BMI": b})
        S = dmatrix(
            f"bs(gest_weeks, df={self.k}, degree=3, include_intercept=False)",
            data=df_new, return_type="dataframe"
        )
        S.columns = [f"s{i+1}" for i in range(S.shape[1])]
        Xg = S.copy()
        Xg["BMI"] = df_new["BMI"].values
        if self.use_tensor_interact:
            for c in S.columns:
                col = f"{c}:BMI"
                if col in self.cols:
                    Xg[col] = Xg[c] * Xg["BMI"]
        Xg = Xg.reindex(columns=self.cols, fill_value=0.0)
        eta = Xg.values @ self.beta
        if self.cov_fe is not None:
            se = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, self.cov_fe, Xg.values))
        else:
            se = np.full_like(eta, np.nan)
        return eta, se

class EnhancedGAMMPredictor:
    def __init__(self, beta, cov_fe, X_columns, spline_df, 
                 use_tensor_interact=False, 
                 default_age=None, default_height=None, default_weight_residual=0.0,
                 default_unique_reads=None, default_gc_content=None):
        self.beta = np.asarray(beta)
        self.cov_fe = np.asarray(cov_fe) if cov_fe is not None else None
        self.cols = list(X_columns)
        self.k = int(spline_df)
        self.use_tensor_interact = use_tensor_interact
        # 默认值（用于预测）
        self.default_age = default_age
        self.default_height = default_height
        self.default_weight_residual = default_weight_residual
        self.default_unique_reads = default_unique_reads
        self.default_gc_content = default_gc_content

    def predict_logit_and_se(self, t, b, age=None, height=None, weight_residual=None, 
                           unique_reads=None, gc_content=None):
        from patsy import dmatrix
        import pandas as pd
        
        # Convert scalar inputs to arrays if needed
        if np.isscalar(t) and np.isscalar(b):
            t = np.array([t])
            b = np.array([b])
            is_scalar = True
        else:
            is_scalar = False
            
        # Ensure both are arrays of the same length
        t = np.atleast_1d(t)
        b = np.atleast_1d(b)
        
        # If different lengths, broadcast to the same length
        if len(t) != len(b):
            if len(t) == 1:
                t = np.full_like(b, t[0])
            elif len(b) == 1:
                b = np.full_like(t, b[0])
                
        # Create DataFrame with explicit index
        df_new = pd.DataFrame({"gest_weeks": t, "BMI": b}, index=range(len(t)))

        S = dmatrix(
            f"bs(gest_weeks, df={self.k}, degree=3, include_intercept=False)",
            data=df_new, return_type="dataframe", eval_env=1
        )
        S.columns = [f"s{i+1}" for i in range(S.shape[1])]
        Xg = S.copy()
        Xg["BMI"] = df_new["BMI"].values
        
        if self.use_tensor_interact:
            for c in S.columns:
                col = f"{c}:BMI"
                if col in self.cols:
                    Xg[col] = Xg[c] * Xg["BMI"]
                
        Xg = Xg.reindex(columns=self.cols, fill_value=0.0)
        eta = Xg.values @ self.beta
        if self.cov_fe is not None:
            se = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, self.cov_fe, Xg.values))
        else:
            se = np.full_like(eta, np.nan)
            
        if is_scalar:
            return eta[0:1], se[0:1]
        else:
            return eta, se
