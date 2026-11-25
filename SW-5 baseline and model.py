"""
SW-5 | Satellite Collision Risk Predictor
Baseline vs model on the ESA Kelvins collision-avoidance CDM data.

Usage:   python "SW-5 baseline and model.py" train_data.csv [cost_ratio]
         (cost_ratio = how much worse a missed event is than a false alarm; default 20)
Needs:   pandas, numpy, scikit-learn  (lightgbm is used if installed)
Makes:   results.csv, predictions.csv, val_predictions.csv, tradeoff.csv, feature_importance.csv, calibration.csv, explanations.csv, model.joblib

Rules baked in:
- features use ONLY CDMs received 2+ days before closest approach (no leakage)
- target = risk of the last CDM of the event ("final risk", log10 scale)
- dangerous = final risk >= 1e-6 (log10 >= -6), the ESA challenge definition
- split is by event order (train 70% / val 15% / test 15%). The public data has no absolute
    timestamps, so call it an "event-ordered split", not a chronological one.
- needs train_data.csv (it contains the final CDM of each event). test_data.csv is cut at
    2 days, has no final risk, and cannot be scored.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    HAVE_LGB = False

DANGER = -6.0          # log10 final risk that counts as dangerous
CUTOFF = 2.0           # days before closest approach
C_MISS, C_FALSE = 20.0, 1.0   # cost of a missed event vs a false alarm (or pass a number as 2nd argument)
EXTRA = ['miss_distance', 'relative_speed', 'mahalanobis_distance', 'max_risk_estimate',
         'max_risk_scaling', 'c_position_covariance_det', 't_position_covariance_det',
         'c_sigma_r', 'c_sigma_t', 'c_sigma_n', 't_sigma_r', 't_sigma_t', 't_sigma_n']


FRIENDLY = {'last_risk': 'latest reported risk', 'max_risk': 'highest risk seen', 'min_risk': 'lowest risk seen',
            'mean_risk': 'average risk', 'risk_slope': 'risk trend', 'risk_delta': 'last change in risk',
            'n_cdm': 'number of CDMs', 'span_days': 'length of message history', 'last_tca': 'time of latest CDM',
            'floor_share': 'share of near-zero risk reports', 'last_is_floor': 'latest report near zero'}


def friendly(name):
    base = name.replace('_last', '').replace('_slope', '')
    if name in FRIENDLY:
        return FRIENDLY[name]
    return base.replace('_', ' ') + (' trend' if name.endswith('_slope') else '')


def explain(model, X, feats, k=2):
    """Per-event reasons from LightGBM's built-in contributions (SHAP-style). Returns two lists of text."""
    c = model.predict(X, pred_contrib=True)[:, :-1]
    up, down = [], []
    for row in c:
        order = np.argsort(row)
        up.append('; '.join(friendly(feats[i]) for i in order[::-1][:k] if row[i] > 0))
        down.append('; '.join(friendly(feats[i]) for i in order[:k] if row[i] < 0))
    return up, down


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def slope(t, y):
    return float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 and np.ptp(t) > 0 else 0.0


def build(df):
    extra = [c for c in EXTRA if c in df.columns]
    rows = []
    for eid, g in df.groupby('event_id'):
        g = g.sort_values('time_to_tca', ascending=False)
        final = g['risk'].iloc[-1]                      # last CDM = final risk
        v = g[g['time_to_tca'] >= CUTOFF]               # only what is visible 2+ days out
        if v.empty:
            continue
        t, r = v['time_to_tca'].to_numpy(), v['risk'].to_numpy()
        f = {'event_id': eid, 'final_risk': final, 'dangerous': final >= DANGER,
             'last_risk': r[-1], 'max_risk': r.max(), 'min_risk': r.min(), 'mean_risk': r.mean(),
             'risk_delta': r[-1] - r[-2] if len(r) > 1 else 0.0, 'risk_slope': slope(t, r),
             'n_cdm': len(v), 'span_days': float(t[0] - t[-1]), 'last_tca': float(t[-1]),
             'floor_share': float((r <= -29.9).mean()), 'last_is_floor': float(r[-1] <= -29.9)}
        for c in extra:
            x = v[c].to_numpy(dtype=float)
            f[c + '_last'] = x[-1]
            f[c + '_slope'] = slope(t, x)
        rows.append(f)
    return pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)


def counts(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    return int((y & ~pred).sum()), int((~y & pred).sum())   # missed, false alarms


def curve(y, score):
    """(missed, false alarms) at every possible threshold on a score."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    pos, pts = int(y.sum()), []
    for t in np.unique(score):
        flag = score >= t
        tp = int((y & flag).sum())
        pts.append((pos - tp, int(flag.sum()) - tp))
    return pts


def fa_at(pts, max_missed):
    ok = [fa for mi, fa in pts if mi <= max_missed]
    return min(ok) if ok else np.nan


def show(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:                                      # tabulate not installed
        return df.to_string(index=False)


def main(path, ratio=None):
    c_miss = float(ratio) if ratio else C_MISS
    
    # Safe low-memory reading to prevent Windows NumPy allocation crash
    try:
        df = pd.read_csv(path, low_memory=True)
    except Exception:
        chunks = pd.read_csv(path, chunksize=100000, low_memory=True)
        df = pd.concat(chunks, ignore_index=True)

    need = {'event_id', 'time_to_tca', 'risk'}
    if not need <= set(df.columns):
        sys.exit(f'Missing columns: {need - set(df.columns)}. Found: {list(df.columns)[:15]}...')
    if (df.groupby('event_id')['time_to_tca'].min() >= CUTOFF).all():
        sys.exit('This file is cut at 2 days (no final CDM per event), so it has no labels to score.\n'
                 'Use train_data.csv from Kelvins instead.')
    ev = build(df)
    feats = [c for c in ev.columns if c not in ('event_id', 'final_risk', 'dangerous')]
    n = len(ev)
    a, b = int(n * 0.70), int(n * 0.85)
    tr, va, te = ev.iloc[:a], ev.iloc[a:b], ev.iloc[b:]
    print('NOTE: split is by event order; the public data has no timestamps, so do not call it chronological.')
    print(f'events: {n} (train {len(tr)}, val {len(va)}, test {len(te)}); '
          f'dangerous in test: {int(te.dangerous.sum())}')
    if te.dangerous.sum() < 10:
        print('WARNING: very few dangerous events in test; treat numbers as rough.')

    pos, neg = tr.dangerous.sum(), (~tr.dangerous).sum()
    if HAVE_LGB:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                               scale_pos_weight=neg / max(pos, 1), random_state=0, verbose=-1)
    else:
        m = HistGradientBoostingClassifier(class_weight='balanced', random_state=0)
    m.fit(tr[feats], tr.dangerous)
    p_va, p_te = m.predict_proba(va[feats])[:, 1], m.predict_proba(te[feats])[:, 1]
    # Platt scaling on VALIDATION: class weights inflate raw probabilities, so rescale them to honest ones
    platt = LogisticRegression(C=1e6).fit(logit(p_va), va.dangerous)
    p_va, p_te = platt.predict_proba(logit(p_va))[:, 1], platt.predict_proba(logit(p_te))[:, 1]

    # pick the decision threshold on VALIDATION by expected cost, then apply to TEST
    grid = np.linspace(0.02, 0.98, 49)
    cost = [c_miss * counts(va.dangerous, p_va >= t)[0] + C_FALSE * counts(va.dangerous, p_va >= t)[1] for t in grid]
    thr = float(grid[int(np.argmin(cost))])
    lo = thr / 3.0

    out = []
    for name, thr_r in (('Baseline (latest risk >= 1e-6)', -6.0), ('Baseline (latest risk >= 1e-4)', -4.0)):
        mi, fa = counts(te.dangerous, te.last_risk >= thr_r)
        out.append((name, mi, fa, average_precision_score(te.dangerous, te.last_risk)))
    mi, fa = counts(te.dangerous, p_te >= thr)
    out.append((f'Our model (threshold {thr:.2f})', mi, fa, average_precision_score(te.dangerous, p_te)))
    res = pd.DataFrame(out, columns=['Method', 'Missed', 'False alarms', 'PR-AUC']).round({'PR-AUC': 3})
    print('\n' + show(res))

    # fair comparison across ALL thresholds: false alarms needed to keep misses under a limit
    bc, mc = curve(te.dangerous, te.last_risk), curve(te.dangerous, p_te)
    trade = pd.DataFrame({'Allowed missed': [0, 3, 5, 10, 20],
                          'Baseline false alarms': [fa_at(bc, k) for k in (0, 3, 5, 10, 20)],
                          'Model false alarms': [fa_at(mc, k) for k in (0, 3, 5, 10, 20)]})
    print('\nFalse alarms needed to keep missed events at or below a limit (all thresholds, test events):\n' + trade.to_string(index=False))
    trade.to_csv('tradeoff.csv', index=False)

    # Manoeuvre / Defer / No manoeuvre
    dec = np.where(p_te >= thr, 'Manoeuvre', np.where(p_te >= lo, 'Defer', 'No manoeuvre'))
    te = te.assign(prob=p_te, decision=dec)
    table = te.groupby('decision').agg(events=('dangerous', 'size'), dangerous=('dangerous', 'sum'))
    print('\nDecision outcomes on the test events (manoeuvre >= %.2f, defer >= %.2f):' % (thr, lo))
    print(table.reindex(['Manoeuvre', 'Defer', 'No manoeuvre']).fillna(0).astype(int).to_string())
    print('Missed dangerous events = "dangerous" in the No manoeuvre row; Defer events wait for the next CDM.')

    # calibration: does "30% danger" really mean about 30%?
    bins = [0, .05, .1, .25, .5, 1.0001]
    te['bin'] = pd.cut(te.prob, bins, right=False)
    cal = te.groupby('bin', observed=True).agg(events=('dangerous', 'size'), predicted=('prob', 'mean'),
                                               observed=('dangerous', 'mean')).round(3).reset_index()
    cal['bin'] = cal['bin'].astype(str)
    print('\nCalibration (predicted vs observed share of dangerous events):\n' + cal.to_string(index=False))
    print(f'\nBrier score (model): {brier_score_loss(te.dangerous, p_te):.4f}')

    # which features matter
    if HAVE_LGB:
        imp = m.feature_importances_.astype(float)
    else:
        imp = permutation_importance(m, va[feats], va.dangerous, n_repeats=3, scoring='average_precision',
                                     random_state=0).importances_mean
    fi = pd.DataFrame({'feature': feats, 'importance': imp}).sort_values('importance', ascending=False)
    print('\nTop features:\n' + fi.head(8).to_string(index=False))
    print(f'\nCost ratio used: missed = {c_miss:g} x false alarm. Model: {"LightGBM" if HAVE_LGB else "sklearn HistGradientBoosting"}')

    if HAVE_LGB:
        try:
            up, down = explain(m, te[feats], feats)
            pd.DataFrame({'event_id': te.event_id.to_numpy(), 'toward_manoeuvre': up,
                          'toward_no_manoeuvre': down}).to_csv('explanations.csv', index=False)
        except Exception as ex:                      # explanations are a bonus, never block the run
            print('Skipped explanations:', ex)
    res.to_csv('results.csv', index=False)
    cal.to_csv('calibration.csv', index=False)
    fi.to_csv('feature_importance.csv', index=False)
    te[['event_id', 'final_risk', 'dangerous', 'last_risk', 'risk_slope', 'n_cdm', 'prob', 'decision']].to_csv('predictions.csv', index=False)
    va.assign(prob=p_va)[['event_id', 'dangerous', 'prob']].to_csv('val_predictions.csv', index=False)
    joblib.dump({'model': m, 'platt': platt, 'features': feats, 'thr': thr, 'lo': lo}, 'model.joblib')
    print('\nSaved: results.csv, calibration.csv, feature_importance.csv, predictions.csv, val_predictions.csv, explanations.csv (if LightGBM), model.joblib')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'train_data.csv', sys.argv[2] if len(sys.argv) > 2 else None)