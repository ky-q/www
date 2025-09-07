import numpy as np
import pandas as pd
from scipy.stats import norm
from config.constant import CFG
def convert_pregnancy_week(week_str: str) -> float:
    if week_str is None or (isinstance(week_str, float) and np.isnan(week_str)):
        return np.nan
    s = str(week_str).strip().lower()
    if "w+" in s:
        w, d = s.split("w+")
        weeks = float(w)
        days = float(d) if d != "" else 0.0
    else:
        weeks = float(s.replace("w", ""))
        days = 0.0
    return weeks + days / 7.0
def logit_clip(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
def drop_near_constant_and_collinear(X: pd.DataFrame, prefer_drop_prefix=("s",), tol_std=1e-10):
    keep = list(X.columns)
    for c in list(keep):
        if np.nanstd(X[c].values) < tol_std:
            keep.remove(c)
            print(f"[clean] drop near-constant col: {c}")
    def _prefer(col):
        return any(col.startswith(p) for p in prefer_drop_prefix)
    while np.linalg.matrix_rank(X[keep].values) < len(keep):
        A = X[keep].values
        corr = np.corrcoef(A, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        i, j = np.unravel_index(np.nanargmax(np.abs(corr)), corr.shape)
        c1, c2 = keep[i], keep[j]
        cand = sorted([c1, c2], key=lambda c: (not _prefer(c), X[c].std()))
        drop_col = cand[0]
        keep.remove(drop_col)
        print(f"[collinear] drop {drop_col} (|corr|={abs(corr[i,j]):.6f})")
    return X[keep]
def logistic(x): return 1.0 / (1.0 + np.exp(-x))
def z_from_conf(level): return norm.ppf(level)
def piecewise_risk(t):
    if t <= 12:
        return CFG.RISK_EARLY
    elif t <= 27:
        return CFG.RISK_MID
    else:
        return CFG.RISK_LATE
def pava_monotone_increasing(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    xi, yi = x[mask], y[mask]
    n = len(yi)
    if n == 0:
        return y
    y_hat = yi.copy()
    w = np.ones_like(yi)
    i = 0
    while i < n - 1:
        if y_hat[i] <= y_hat[i + 1]:
            i += 1
        else:
            s = w[i] * y_hat[i] + w[i + 1] * y_hat[i + 1]
            ww = w[i] + w[i + 1]
            y_hat[i] = y_hat[i + 1] = s / ww
            w[i] = w[i + 1] = ww
            j = i
            while j > 0 and y_hat[j - 1] > y_hat[j]:
                s = w[j - 1] * y_hat[j - 1] + w[j] * y_hat[j]
                ww = w[j - 1] + w[j]
                y_hat[j - 1] = y_hat[j] = s / ww
                w[j - 1] = w[j] = ww
                j -= 1
            i = max(j, 0)
    out = y.copy()
    out[mask] = y_hat
    return out
def smooth_ma(y, k=5):
    k = max(1, int(k))
    if k == 1:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")
