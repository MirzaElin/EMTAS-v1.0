
import math
import numpy as np
from collections import Counter, defaultdict

# ----------------------- Binary classification metrics -----------------------

def _safe_div(num, den):
    return float(num) / float(den) if den else float("nan")

def confusion_counts(y_true, y_pred):
    tp = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==0)
    return tp, fp, tn, fn

def wilson_ci(successes, n, z=1.96):
    # Wilson score interval for binomial proportion
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / denom
    margin = (z * math.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

def binary_metrics(y_true, y_pred):
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    n = tp+fp+tn+fn
    acc = _safe_div(tp+tn, n)
    tpr = _safe_div(tp, tp+fn)
    tnr = _safe_div(tn, tn+fp)
    ppv = _safe_div(tp, tp+fp)
    npv = _safe_div(tn, tn+fn)
    f1 = _safe_div(2*ppv*tpr, (ppv+tpr)) if not math.isnan(ppv) and not math.isnan(tpr) else float("nan")
    balacc = (tpr + tnr)/2 if not math.isnan(tpr) and not math.isnan(tnr) else float("nan")
    youden = (tpr + tnr - 1) if not math.isnan(tpr) and not math.isnan(tnr) else float("nan")
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn)/denom if denom else float("nan")
    # CIs for proportions
    acc_ci = wilson_ci(tp+tn, n) if n else (float("nan"), float("nan"))
    sens_ci = wilson_ci(tp, tp+fn) if (tp+fn)>0 else (float("nan"), float("nan"))
    spec_ci = wilson_ci(tn, tn+fp) if (tn+fp)>0 else (float("nan"), float("nan"))
    ppv_ci = wilson_ci(tp, tp+fp) if (tp+fp)>0 else (float("nan"), float("nan"))
    npv_ci = wilson_ci(tn, tn+fn) if (tn+fn)>0 else (float("nan"), float("nan"))
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "N": n,
        "Accuracy": acc, "Accuracy_CI": acc_ci,
        "Sensitivity_TPR": tpr, "Sensitivity_CI": sens_ci,
        "Specificity_TNR": tnr, "Specificity_CI": spec_ci,
        "PPV": ppv, "PPV_CI": ppv_ci,
        "NPV": npv, "NPV_CI": npv_ci,
        "F1": f1, "Balanced_Accuracy": balacc, "Youdens_J": youden, "MCC": mcc
    }

def roc_points_auc(y_true, scores):
    # Compute ROC by sweeping unique thresholds
    # Returns FPR, TPR arrays and AUC (trapezoidal)
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    P = sum(y_true)
    N = len(y_true) - P
    tps, fps = 0, 0
    roc = [(0.0, 0.0)]
    prev_score = None
    i = 0
    while i < len(pairs):
        thr = pairs[i][0]
        # Move through ties
        tp_inc = 0
        fp_inc = 0
        while i < len(pairs) and pairs[i][0] == thr:
            if pairs[i][1] == 1:
                tp_inc += 1
            else:
                fp_inc += 1
            i += 1
        tps += tp_inc
        fps += fp_inc
        fpr = fps / N if N else 0.0
        tpr = tps / P if P else 0.0
        roc.append((fpr, tpr))
    roc.append((1.0, 1.0))
    # AUC
    xs, ys = zip(*roc)
    auc = 0.0
    for j in range(1, len(xs)):
        dx = xs[j] - xs[j-1]
        auc += dx * (ys[j] + ys[j-1]) / 2.0
    return np.array(xs), np.array(ys), auc

def pr_points_ap(y_true, scores):
    # Precision-Recall curve + Average Precision (area under PR step function)
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    tp, fp = 0, 0
    P = sum(y_true)
    precisions = []
    recalls = []
    last_recall = -1
    for score, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / P if P else 0.0
        # only keep points where recall increases
        if rec != last_recall:
            precisions.append(prec); recalls.append(rec)
            last_recall = rec
    # AP via trapezoidal on (recall, precision) with sorting by recall
    if not recalls:
        return np.array([0.0,1.0]), np.array([1.0,0.0]), 0.0
    r = np.array(recalls)
    p = np.array(precisions)
    order = np.argsort(r)
    r = r[order]; p = p[order]
    ap = 0.0
    for i in range(1, len(r)):
        ap += (r[i]-r[i-1]) * ((p[i]+p[i-1])/2)
    return r, p, ap

# ----------------------------- Fleiss' kappa ---------------------------------

def fleiss_kappa_from_raw(ratings_matrix):
    """
    ratings_matrix: list of lists (n_items x n_raters) with nominal categories (hashable)
    Returns: kappa, per_item_Pi list, category_marginals dict
    """
    n_items = len(ratings_matrix)
    if n_items == 0:
        return float("nan"), [], {}
    n_raters = len(ratings_matrix[0])
    categories = sorted(set(c for row in ratings_matrix for c in row if c is not None))
    cat_index = {c:i for i,c in enumerate(categories)}
    # counts per item x category
    counts = np.zeros((n_items, len(categories)), dtype=int)
    for i,row in enumerate(ratings_matrix):
        if len(row) != n_raters:
            raise ValueError("All items must have the same number of raters")
        for c in row:
            if c is None or c not in cat_index:
                continue
            counts[i, cat_index[c]] += 1
    # per-item agreement
    P_i = []
    for i in range(n_items):
        n_i = counts[i].sum()
        if n_i == 0:
            P_i.append(float("nan"))
        else:
            P_i.append((np.sum(counts[i]*counts[i]) - n_i) / (n_i*(n_i-1)))
    P_bar = np.nanmean(P_i)
    # category proportions
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j*p_j)
    kappa = (P_bar - P_e) / (1 - P_e) if (1-P_e)!=0 else float("nan")
    marginals = {cat: float(p) for cat,p in zip(categories, p_j)}
    return float(kappa), [float(x) for x in P_i], marginals

def bootstrap_fleiss_ci(ratings_matrix, B=500, seed=123):
    rng = np.random.default_rng(seed)
    n = len(ratings_matrix)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [ratings_matrix[i] for i in idx]
        kappa, _, _ = fleiss_kappa_from_raw(sample)
        if not math.isnan(kappa):
            stats.append(kappa)
    if not stats:
        return (float("nan"), float("nan"))
    low = np.percentile(stats, 2.5)
    high = np.percentile(stats, 97.5)
    return (float(low), float(high))

# --------------------------------- ICC(2,1) ----------------------------------

def icc2_1(matrix):
    """
    matrix: numpy array (n_subjects x k_raters), numeric
    Returns: ICC(2,1) and variance components
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("Matrix must be 2D")
    n, k = X.shape
    # Means
    grand = np.nanmean(X)
    mean_rows = np.nanmean(X, axis=1)
    mean_cols = np.nanmean(X, axis=0)
    # Sum squares (nan-safe: replace nan with row/col mean where needed)
    X_filled = X.copy()
    # impute missing with row means (fallback to grand)
    for i in range(n):
        row = X_filled[i]
        m = np.nanmean(row) if not np.isnan(np.nanmean(row)) else grand
        row[np.isnan(row)] = m
        X_filled[i] = row
    grand = np.mean(X_filled)
    mean_rows = np.mean(X_filled, axis=1)
    mean_cols = np.mean(X_filled, axis=0)
    SSR = k * np.sum((mean_rows - grand)**2)
    SSC = n * np.sum((mean_cols - grand)**2)
    SSE = np.sum((X_filled - mean_rows[:,None] - mean_cols[None,:] + grand)**2)
    MSR = SSR / (n-1) if n>1 else float("nan")
    MSC = SSC / (k-1) if k>1 else float("nan")
    MSE = SSE / ((n-1)*(k-1)) if (n>1 and k>1) else float("nan")
    icc = (MSR - MSE) / (MSR + (k-1)*MSE + (k*(MSC - MSE))/n) if n>1 and k>1 else float("nan")
    components = {
        "MSR": float(MSR), "MSC": float(MSC), "MSE": float(MSE),
        "SSR": float(SSR), "SSC": float(SSC), "SSE": float(SSE),
        "n_subjects": int(n), "k_raters": int(k), "grand_mean": float(grand)
    }
    return float(icc), components

def bootstrap_icc_ci(matrix, B=1000, seed=123):
    X = np.array(matrix, dtype=float)
    n, k = X.shape
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = X[idx]
        icc, _ = icc2_1(sample)
        if not math.isnan(icc):
            stats.append(icc)
    if not stats:
        return (float("nan"), float("nan"))
    low = float(np.percentile(stats, 2.5))
    high = float(np.percentile(stats, 97.5))
    return (low, high)


def bootstrap_icc_distribution(matrix, B=1000, seed=123):
    """
    Return the bootstrap distribution of ICC(2,1) values to enable
    transforming to ICC(2,k) (average-measure) and computing its CI.
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2:
        return np.array([], dtype=float)
    n, k = X.shape
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = X[idx]
        icc, _ = icc2_1(sample)
        if not math.isnan(icc):
            stats.append(icc)
    return np.array(stats, dtype=float)
