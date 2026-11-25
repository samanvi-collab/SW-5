"""
SW-5 dashboard (Streamlit).  Run:  streamlit run app.py
Needs in the same folder: train_data.csv (Kelvins) and the files made by sw5_baseline_and_model.py
(predictions.csv, val_predictions.csv, results.csv, calibration.csv, feature_importance.csv, explanations.csv).
Install: pip install streamlit pandas numpy
"""
import os
import numpy as np
import pandas as pd

CUTOFF = 2.0
ORDER = {'Manoeuvre': 0, 'Defer': 1, 'No manoeuvre': 2}


def decisions(pred, c_miss, val=None):
    """Pick the manoeuvre threshold on VALIDATION events (not the test events), then apply it to the test events."""
    ref = val if val is not None else pred
    y = ref.dangerous.to_numpy(bool)
    p = ref.prob.to_numpy()
    best, best_cost = 0.5, None
    for t in np.linspace(0.02, 0.98, 49):
        miss = int((y & (p < t)).sum())
        fa = int((~y & (p >= t)).sum())
        cost = c_miss * miss + fa
        if best_cost is None or cost < best_cost:
            best, best_cost = float(t), cost
    lo = best / 3.0
    pt = pred.prob.to_numpy()
    dec = np.where(pt >= best, 'Manoeuvre', np.where(pt >= lo, 'Defer', 'No manoeuvre'))
    return pred.assign(decision=dec), best, lo


def outcome(pred):
    """Label each event so you can find examples: correct calls, false alarms, misses, deferrals."""
    y, d = pred.dangerous.astype(bool), pred.decision
    return np.select([(d == 'Manoeuvre') & y, (d == 'Manoeuvre') & ~y, (d == 'No manoeuvre') & y, d == 'Defer'],
                     ['Correct manoeuvre', 'False alarm', 'Missed dangerous', 'Deferred'], 'Correct no manoeuvre')


def ui():
    import streamlit as st
    st.set_page_config(page_title='SW-5 Collision Risk', layout='wide')

    @st.cache_data
    def load():
        raw = pd.read_csv('train_data.csv', usecols=['event_id', 'time_to_tca', 'risk'])
        pred = pd.read_csv('predictions.csv')
        extras = {k: pd.read_csv(k + '.csv') for k in ('results', 'calibration', 'feature_importance', 'explanations', 'val_predictions', 'tradeoff') if os.path.exists(k + '.csv')}
        return raw, pred, extras

    raw, pred0, extras = load()
    st.title('Satellite Collision Risk Predictor')
    st.caption('Team DevX | Inputs: CDMs received 2+ days before closest approach. The final risk is shown only to check the call.')

    c_miss = st.sidebar.slider('A missed event costs this many false alarms', 1, 100, 20)
    pred, thr, lo = decisions(pred0, c_miss, extras.get('val_predictions'))
    pred = pred.assign(rank=pred.decision.map(ORDER)).sort_values(['rank', 'prob'], ascending=[True, False])
    st.sidebar.write(f'Manoeuvre at probability >= {thr:.2f}')
    st.sidebar.write(f'Defer at probability >= {lo:.2f}')
    if 'val_predictions' in extras:
        st.sidebar.caption('Thresholds are picked on validation events, then applied to the test events.')
    else:
        st.sidebar.warning('val_predictions.csv is missing. Rerun sw5_baseline_and_model.py. Until then the thresholds are tuned on test events and are NOT honest.')
    full = pred

    tab_events, tab_results = st.tabs(['Events', 'Results'])
    with tab_events:
        left, right = st.columns([1, 2])
        with left:
            st.subheader('Ranked events')
            pred = pred.assign(outcome=outcome(pred))
            kind = st.radio('Show', ['All', 'Correct manoeuvre', 'False alarm', 'Missed dangerous', 'Deferred'], horizontal=True)
            if kind != 'All':
                pred = pred[pred.outcome == kind]
            if pred.empty:
                st.info('No events in this group.')
                st.stop()
            st.dataframe(pred[['event_id', 'decision', 'outcome', 'prob']].rename(columns={'prob': 'danger probability'}).head(60),
                         hide_index=True, use_container_width=True)
        with right:
            eid = st.selectbox('Event', pred.event_id.tolist())
            row = pred[pred.event_id == eid].iloc[0]
            g = raw[raw.event_id == eid].sort_values('time_to_tca', ascending=False)
            vis = g[g.time_to_tca >= CUTOFF]
            colour = {'Manoeuvre': '#c62828', 'Defer': '#f2a900', 'No manoeuvre': '#2e7d32'}[row.decision]
            st.markdown(f"<h2 style='color:{colour}'>{row.decision.upper()}</h2>", unsafe_allow_html=True)
            st.write(f'Danger probability: **{row.prob:.2f}**')
            chart = vis.assign(day=(-vis.time_to_tca).round(2), risk=vis.risk.clip(lower=-12).round(2))
            st.line_chart(chart.set_index('day')[['risk']].rename(columns={'risk': 'reported risk (log10, floor at -12)'}))
            st.caption('x axis: days relative to closest approach (0 = closest approach). Only CDMs 2+ days out are used.')
            st.subheader('Why')
            ex = extras.get('explanations')
            hit = ex[ex.event_id == eid] if ex is not None else None
            if hit is not None and len(hit):
                h = hit.iloc[0]
                st.write('**Pushing toward manoeuvre:** ' + (h.toward_manoeuvre if isinstance(h.toward_manoeuvre, str) and h.toward_manoeuvre else 'nothing strong'))
                st.write('**Pushing toward no manoeuvre:** ' + (h.toward_no_manoeuvre if isinstance(h.toward_no_manoeuvre, str) and h.toward_no_manoeuvre else 'nothing strong'))
            else:
                st.write('Simple checks: latest risk %.2f, %d CDMs, trend %s' % (
                    getattr(row, 'last_risk', row.prob), 
                    int(getattr(row, 'n_cdm', len(vis))), 
                    'rising' if getattr(row, 'risk_slope', 0) < 0 else 'flat or falling'
                ))
            st.caption('Check against reality (hidden from the model): final risk = %.2f, dangerous = %s'
                       % (getattr(row, 'final_risk', row.prob), bool(row.dangerous)))

    with tab_results:
        if 'results' in extras:
            st.subheader('Baseline vs model (Dynamic Threshold Impact)')
            y_te = full.dangerous.astype(bool)
            mod_missed, mod_fa = int((y_te & (full.decision == 'No manoeuvre')).sum()), int((~y_te & (full.decision == 'Manoeuvre')).sum())
            dyn_results = extras['results'].copy()
            dyn_results.loc[dyn_results['Method'].str.contains('Our model'), ['Missed', 'False alarms']] = [mod_missed, mod_fa]
            st.dataframe(dyn_results, hide_index=True)
        y = full.dangerous.astype(bool)
        d = full.decision
        missed = int((y & (d == 'No manoeuvre')).sum())
        false_alarms = int((~y & (d == 'Manoeuvre')).sum())
        a, b, c = st.columns(3)
        a.metric('Events scored', len(full))
        b.metric('Missed dangerous events', missed)
        c.metric('Unnecessary manoeuvres', false_alarms)
        st.write(full.groupby('decision').agg(events=('dangerous', 'size'), dangerous=('dangerous', 'sum')).reindex(list(ORDER)).fillna(0).astype(int))
        if 'results' in extras:
            st.subheader('Baseline vs model')
            st.dataframe(extras['results'], hide_index=True)
        if 'tradeoff' in extras:
            st.subheader('False alarms needed to keep misses under a limit')
            st.dataframe(extras['tradeoff'], hide_index=True)
        if 'calibration' in extras:
            st.subheader('Calibration')
            st.dataframe(extras['calibration'], hide_index=True)
        if 'feature_importance' in extras:
            st.subheader('Top features')
            fi = extras['feature_importance'].head(10).copy()
            fi['share'] = fi.importance / fi.importance.sum()
            st.dataframe(fi[['feature', 'share']], hide_index=True, use_container_width=True,
                         column_config={'share': st.column_config.ProgressColumn('importance', min_value=0.0, max_value=1.0, format='%.2f')})


if __name__ == '__main__':
    ui()