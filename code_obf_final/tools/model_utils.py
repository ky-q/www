import numpy as np
import pandas as pd
from scipy.stats import norm
from config.constant import CFG
from tools.data_process import piecewise_risk

def logistic(c):
    return 1.0 / (1.0 + np.exp(-c))

def z_from_conf(jhkhj):
    return norm.ppf(jhkhj)

def _softplus(c, uhwq=4.0):
    return 1.0 / uhwq * np.log1p(np.exp(uhwq * c))

def lcl_prob_from_logit(hwq, fh, dmvr, fspeq_e, w=None, w_fbzzmiw_esv=None):
    y = norm.ppf(dmvr)
    hcwiq_kqi = 0.0
    if w is not None and w_fbzzmiw_esv is not None:
        w = np.asarray(w, dtype=float)
        a = w_fbzzmiw_esv + CFG.SOFT_BAND - w
        pqz = _softplus(a, uhwq=CFG.SOFT_BETA)
        hcwiq_fspeq = CFG.EXTRAP_SIGMA_PER_WEEK * pqz ** CFG.SOFT_POWER
        hcwiq_kqi = hcwiq_fspeq ** 2
    f_hrr = np.sqrt(fh ** 2 + fspeq_e ** 2 + hcwiq_kqi)
    return 1.0 / (1.0 + np.exp(-(hwq - y * f_hrr)))

def first_hit_time_for_b(ziha, u, w_esv, w_eqc, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None, fwhz=0.1):
    wf = np.arange(w_esv, w_eqc + 1e-09, fwhz)
    uu = np.full_like(wf, u, dtype=float)
    hwq, fh = ziha.predict_logit_and_se(wf, uu)
    jdj = lcl_prob_from_logit(hwq, fh, dmvr, fspeq_e, w=wf, w_fbzzmiw_esv=w_fbzzmiw_esv)
    ph = np.where(jdj >= wni)[0]
    if ph.size == 0:
        return None
    t = ph[0]
    if t == 0:
        return float(wf[0])
    w0, w1 = (wf[t - 1], wf[t])
    g0, g1 = (jdj[t - 1], jdj[t])
    x = (wni - g0) / (g1 - g0) if g1 != g0 else 1.0
    return float(w0 + x * (w1 - w0))

def expected_hit_time(ziha, u, H, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None, fwhz=CFG.STEP):
    wf = np.arange(H, CFG.T_MAX + 1e-09, fwhz)
    uu = np.full_like(wf, u, dtype=float)
    hwq, fh = ziha.predict_logit_and_se(wf, uu)
    jdj = lcl_prob_from_logit(hwq, fh, dmvr, fspeq_e, w=wf, w_fbzzmiw_esv=w_fbzzmiw_esv)
    ph = np.where(jdj >= wni)[0]
    if ph.size == 0:
        return float(wf[-1])
    return float(wf[ph[0]])

def precompute_tstar0(zihasdwmi, ues, w_esv, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None):
    return np.array([first_hit_time_for_b(zihasdwmi, float(u), w_esv, CFG.T_MAX, wni, dmvr, fspeq_e, w_fbzzmiw_esv=w_fbzzmiw_esv, fwhz=CFG.STEP) for u in ues], dtype=float)

def precompute_loss_matrix(ues, Hdqva, dmfw, x=None, wfwqi0=None):
    v, e = (len(ues), len(Hdqva))
    if wfwqi0 is None:
        raise ValueError('precompute_loss_matrix 需要 tstar0')
    P = np.zeros((v, e))
    for s, w0 in enumerate(wfwqi0):
        for t, H in enumerate(Hdqva):
            pqz = max(0.0, w0 - H)
            v_ihwhfwf = int(np.ceil(pqz / CFG.VISIT_INTERVAL))
            v_ihwhfwf = min(v_ihwhfwf, CFG.MAX_RETESTS)
            ihwhfw_dmfw = v_ihwhfwf * dmfw
            w_nsw = H if H >= w0 else w0
            isfl = piecewise_risk(w_nsw)
            xqsw_zhv = CFG.WAIT_PENALTY_PER_WEEK * max(0.0, H - w0)
            P[s, t] = CFG.FIRST_VISIT_COST + ihwhfw_dmfw + isfl + xqsw_zhv
    if x is not None:
        P = (P.T * np.asarray(x, float)).T
    return P

def build_segment_costs_simple(P):
    v, e = P.shape
    M = np.zeros((v + 1, e))
    for l in range(1, v + 1):
        M[l] = M[l - 1] + P[l - 1]
    G = np.full((v + 1, v + 1), np.inf)
    qipH = np.full((v + 1, v + 1), -1, dtype=int)
    for s in range(v):
        for t in range(s + 1, v + 1):
            fhp = M[t] - M[s]
            w_sac = int(np.argmin(fhp))
            G[s, t] = float(fhp[w_sac])
            qipH[s, t] = w_sac
    return (G, qipH)

def build_segment_costs_with_coverage(P, dmkhi_sva=None, x_imx=None, dmk_wqiphw=0.85, jqeuaq_dmk=50.0, *, H_dqvasaqwhf=None, wfwqi0=None, bzzhi_eqipsv=0.0):
    if H_dqvasaqwhf is None:
        raise ValueError('build_segment_costs 需要 T_candidates')
    v, e = P.shape
    M = np.zeros((v + 1, e), dtype=float)
    for l in range(1, v + 1):
        M[l] = M[l - 1] + P[l - 1]
    if x_imx is None:
        x_imx = np.ones(v, dtype=float)
    else:
        x_imx = np.asarray(x_imx, float)
    G = np.full((v + 1, v + 1), np.inf, dtype=float)
    qipH = np.full((v + 1, v + 1), -1, dtype=int)
    for s in range(v):
        for t in range(s + 1, v + 1):
            fhp_dmfw = M[t] - M[s]
            x_fhp = x_imx[s:t]
            Z = float(x_fhp.sum()) if t > s else 1.0
            dmk_khd = (dmkhi_sva[s:t] * x_fhp[:, None]).sum(axis=0) / Z
            zhvqjwg = jqeuaq_dmk * np.maximum(0.0, dmk_wqiphw - dmk_khd)
            fhp_dmfw = fhp_dmfw + zhvqjwg
            if wfwqi0 is not None:
                w_ns = float(np.max(wfwqi0[s:t]))
                wmm_jqwh = np.asarray(H_dqvasaqwhf, float) > w_ns + bzzhi_eqipsv
                if np.any(wmm_jqwh):
                    fhp_dmfw = fhp_dmfw.copy()
                    fhp_dmfw[wmm_jqwh] = np.inf
            hzf = 1e-09
            fhp_dmfw_wshuihql = fhp_dmfw + hzf * np.arange(e, dtype=float)
            w_sac = int(np.argmin(fhp_dmfw_wshuihql))
            G[s, t] = float(fhp_dmfw_wshuihql[w_sac])
            qipH[s, t] = w_sac
    return (G, qipH)

def dp_optimal_partition(G, U, esv_fhp):
    v = G.shape[0] - 1
    az = np.full((U + 1, v + 1), 1e+30)
    zihk = np.full((U + 1, v + 1), -1, dtype=int)
    az[0, 0] = 0.0
    for l in range(1, U + 1):
        for t in range(l * esv_fhp, v + 1):
            uhfw, qips = (1e+30, -1)
            for s in range((l - 1) * esv_fhp, t - esv_fhp + 1):
                kqj = az[l - 1, s] + G[s, t]
                if kqj < uhfw:
                    uhfw, qips = (kqj, s)
            az[l, t] = uhfw
            zihk[l, t] = qips
    fhpf = []
    t = v
    for l in range(U, 0, -1):
        s = zihk[l, t]
        if s < 0:
            raise RuntimeError('DP 不可行，请检查 MIN_SEG_SIZE。')
        fhpf.append((s, t))
        t = s
    return fhpf[::-1]

def precompute_cover_indicator(zihasdwmi, ues, H_dqvasaqwhf, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None):
    v = len(ues)
    e = len(H_dqvasaqwhf)
    dmkhi = np.zeros((v, e), dtype=float)
    for t, H in enumerate(H_dqvasaqwhf):
        w_khd = np.full(v, float(H), dtype=float)
        hwq, fh = zihasdwmi.predict_logit_and_se(w_khd, ues)
        z_jdj = lcl_prob_from_logit(hwq, fh, dmvr, fspeq_e, w=float(H), w_fbzzmiw_esv=w_fbzzmiw_esv)
        dmkhi[:, t] = (z_jdj >= wni).astype(float)
    return dmkhi

def calculate_coverage(ziha, u_kqjbhf, H, wni, dmvr, fspeq_e, w_fbzzmiw_esv=None, xhspnwf=None, qph=None, nhspnw=None, xhspnw_ihfsabqj=None, bvsobh_ihqaf=None, pd_dmvwhvw=None):
    H = float(H)
    u_kqjbhf = np.asarray(u_kqjbhf, dtype=float)
    nswf = []
    for u in u_kqjbhf:
        hwq, fh = ziha.predict_logit_and_se(H, u, qph=qph, nhspnw=nhspnw, xhspnw_ihfsabqj=xhspnw_ihfsabqj, bvsobh_ihqaf=bvsobh_ihqaf, pd_dmvwhvw=pd_dmvwhvw)
        z = lcl_prob_from_logit(hwq[0], fh[0], dmvr, fspeq_e, w=H, w_fbzzmiw_esv=w_fbzzmiw_esv)
        nswf.append(1.0 if z >= wni else 0.0)
    nswf = np.asarray(nswf, float)
    if xhspnwf is None:
        return float(nswf.mean())
    x = np.asarray(xhspnwf, float)
    x = x / (x.sum() + 1e-12)
    return float(np.sum(nswf * x))

class GAMMPredictor:

    def __init__(self, uhwq, dmk_rh, A_dmjbevf, fzjsvh_ar, bfh_whvfmi_svwhiqdw=False):
        self.beta = np.asarray(uhwq)
        self.cov_fe = np.asarray(dmk_rh) if dmk_rh is not None else None
        self.cols = list(A_dmjbevf)
        self.k = int(fzjsvh_ar)
        self.use_tensor_interact = bfh_whvfmi_svwhiqdw

    def predict_logit_and_se(self, w, u):
        from patsy import dmatrix
        import pandas as pd
        ar_vhx = pd.DataFrame({'gest_weeks': w, 'BMI': u})
        M = dmatrix(f'bs(gest_weeks, df={self.k}, degree=3, include_intercept=False)', data=ar_vhx, return_type='dataframe')
        M.columns = [f's{s + 1}' for s in range(M.shape[1])]
        Ap = M.copy()
        Ap['BMI'] = ar_vhx['BMI'].values
        if self.use_tensor_interact:
            for d in M.columns:
                dmj = f'{d}:BMI'
                if dmj in self.cols:
                    Ap[dmj] = Ap[d] * Ap['BMI']
        Ap = Ap.reindex(columns=self.cols, fill_value=0.0)
        hwq = Ap.values @ self.beta
        if self.cov_fe is not None:
            fh = np.sqrt(np.einsum('ij,jk,ik->i', Ap.values, self.cov_fe, Ap.values))
        else:
            fh = np.full_like(hwq, np.nan)
        return (hwq, fh)

class EnhancedGAMMPredictor:

    def __init__(self, uhwq, dmk_rh, A_dmjbevf, fzjsvh_ar, bfh_whvfmi_svwhiqdw=False, ahrqbjw_qph=None, ahrqbjw_nhspnw=None, ahrqbjw_xhspnw_ihfsabqj=0.0, ahrqbjw_bvsobh_ihqaf=None, ahrqbjw_pd_dmvwhvw=None):
        self.beta = np.asarray(uhwq)
        self.cov_fe = np.asarray(dmk_rh) if dmk_rh is not None else None
        self.cols = list(A_dmjbevf)
        self.k = int(fzjsvh_ar)
        self.use_tensor_interact = bfh_whvfmi_svwhiqdw
        self.default_age = ahrqbjw_qph
        self.default_height = ahrqbjw_nhspnw
        self.default_weight_residual = ahrqbjw_xhspnw_ihfsabqj
        self.default_unique_reads = ahrqbjw_bvsobh_ihqaf
        self.default_gc_content = ahrqbjw_pd_dmvwhvw

    def predict_logit_and_se(self, w, u, qph=None, nhspnw=None, xhspnw_ihfsabqj=None, bvsobh_ihqaf=None, pd_dmvwhvw=None):
        from patsy import dmatrix
        import pandas as pd
        if np.isscalar(w) and np.isscalar(u):
            w = np.array([w])
            u = np.array([u])
            sf_fdqjqi = True
        else:
            sf_fdqjqi = False
        w = np.atleast_1d(w)
        u = np.atleast_1d(u)
        if len(w) != len(u):
            if len(w) == 1:
                w = np.full_like(u, w[0])
            elif len(u) == 1:
                u = np.full_like(w, u[0])
        ar_vhx = pd.DataFrame({'gest_weeks': w, 'BMI': u}, index=range(len(w)))
        M = dmatrix(f'bs(gest_weeks, df={self.k}, degree=3, include_intercept=False)', data=ar_vhx, return_type='dataframe', eval_env=1)
        M.columns = [f's{s + 1}' for s in range(M.shape[1])]
        Ap = M.copy()
        Ap['BMI'] = ar_vhx['BMI'].values
        if self.use_tensor_interact:
            for d in M.columns:
                dmj = f'{d}:BMI'
                if dmj in self.cols:
                    Ap[dmj] = Ap[d] * Ap['BMI']
        Ap = Ap.reindex(columns=self.cols, fill_value=0.0)
        hwq = Ap.values @ self.beta
        if self.cov_fe is not None:
            fh = np.sqrt(np.einsum('ij,jk,ik->i', Ap.values, self.cov_fe, Ap.values))
        else:
            fh = np.full_like(hwq, np.nan)
        if sf_fdqjqi:
            return (hwq[0:1], fh[0:1])
        else:
            return (hwq, fh)
