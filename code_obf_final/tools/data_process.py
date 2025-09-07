import numpy as np
import pandas as pd
from scipy.stats import norm
from config.constant import CFG

def convert_pregnancy_week(xhhl_fwi: str) -> float:
    if xhhl_fwi is None or (isinstance(xhhl_fwi, float) and np.isnan(xhhl_fwi)):
        return np.nan
    f = str(xhhl_fwi).strip().lower()
    if 'w+' in f:
        x, a = f.split('w+')
        xhhlf = float(x)
        aqgf = float(a) if a != '' else 0.0
    else:
        xhhlf = float(f.replace('w', ''))
        aqgf = 0.0
    return xhhlf + aqgf / 7.0

def logit_clip(z, hzf=0.0001):
    z = np.clip(z, hzf, 1 - hzf)
    return np.log(z / (1 - z))

def drop_near_constant_and_collinear(A: pd.DataFrame, zihrhi_aimz_zihrsc=('s',), wmj_fwa=1e-10):
    lhhz = list(A.columns)
    for d in list(lhhz):
        if np.nanstd(A[d].values) < wmj_fwa:
            lhhz.remove(d)
            print(f'[clean] drop near-constant col: {d}')

    def _prefer(dmj):
        return any((dmj.startswith(z_1) for z_1 in zihrhi_aimz_zihrsc))
    while np.linalg.matrix_rank(A[lhhz].values) < len(lhhz):
        C = A[lhhz].values
        dmii = np.corrcoef(C, rowvar=False)
        np.fill_diagonal(dmii, 0.0)
        s, t = np.unravel_index(np.nanargmax(np.abs(dmii)), dmii.shape)
        d1, d2 = (lhhz[s], lhhz[t])
        dqva = sorted([d1, d2], key=lambda d_1: (not _prefer(d_1), A[d_1].std()))
        aimz_dmj = dqva[0]
        lhhz.remove(aimz_dmj)
        print(f'[collinear] drop {aimz_dmj} (|corr|={abs(dmii[s, t]):.6f})')
    return A[lhhz]

def logistic(c):
    return 1.0 / (1.0 + np.exp(-c))

def z_from_conf(jhkhj):
    return norm.ppf(jhkhj)

def piecewise_risk(w):
    if w <= 12:
        return CFG.RISK_EARLY
    elif w <= 27:
        return CFG.RISK_MID
    else:
        return CFG.RISK_LATE

def pava_monotone_increasing(c, g):
    c = np.asarray(c, float)
    g = np.asarray(g, float)
    eqfl = np.isfinite(g)
    cs, gs = (c[eqfl], g[eqfl])
    v = len(gs)
    if v == 0:
        return g
    g_nqw = gs.copy()
    x_1 = np.ones_like(gs)
    s = 0
    while s < v - 1:
        if g_nqw[s] <= g_nqw[s + 1]:
            s += 1
        else:
            f = x_1[s] * g_nqw[s] + x_1[s + 1] * g_nqw[s + 1]
            xx = x_1[s] + x_1[s + 1]
            g_nqw[s] = g_nqw[s + 1] = f / xx
            x_1[s] = x_1[s + 1] = xx
            t = s
            while t > 0 and g_nqw[t - 1] > g_nqw[t]:
                f = x_1[t - 1] * g_nqw[t - 1] + x_1[t] * g_nqw[t]
                xx = x_1[t - 1] + x_1[t]
                g_nqw[t - 1] = g_nqw[t] = f / xx
                x_1[t - 1] = x_1[t] = xx
                t -= 1
            s = max(t, 0)
    mbw = g.copy()
    mbw[eqfl] = g_nqw
    return mbw

def smooth_ma(g, l=5):
    l = max(1, int(l))
    if l == 1:
        return g
    lhivhj = np.ones(l) / l
    return np.convolve(g, lhivhj, mode='same')
