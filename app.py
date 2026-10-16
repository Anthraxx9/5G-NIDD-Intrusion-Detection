"""Dashboard: python experiment.py first, then  streamlit run app.py"""
import json
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
import config as C

st.set_page_config(page_title="5G-NIDD anomaly detection", layout="wide")
st.title("Flow-level anomaly detection on 5G-NIDD (offline analysis)")
st.caption("Offline flow-level evaluation. Choose a model; thresholds are calibrated on benign validation flows.")

if not (C.ARTIFACTS / "model.joblib").exists():
    st.error("No trained model found. Run `python experiment.py` first."); st.stop()

bundle = joblib.load(C.ARTIFACTS / "model.joblib")
M = json.loads((C.ARTIFACTS / "metrics.json").read_text())
feats = bundle["num_cols"] + bundle["cat_cols"]

model_choice = st.sidebar.radio("Detection model", ["Isolation Forest (anomaly detection)", "Random Forest (supervised)"], index=0)
use_rf = model_choice.startswith("Random Forest") and "rf" in bundle and "supervised_rf" in M
section = M["supervised_rf"] if use_rf else M
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test ROC-AUC", f"{section['roc_auc']:.3f}")
c2.metric("Test PR-AUC", f"{section['pr_auc']:.3f}", help=f"A coin flip scores {M['test_attack_share']:.3f} here (the attack share).")
c3.metric("Features used", M["n_features"])
c4.metric("Split", M["split_mode"])
if M["leak_warnings"]:
    st.warning("Possible leakage: " + ", ".join(w["feature"] for w in M["leak_warnings"]))
fpr_keys = list(section["thresholds"].keys())
key = st.sidebar.select_slider("Operating point (target false-positive rate)", options=fpr_keys,
                               value=f"fpr_{M['target_fpr']}", format_func=lambda k: f"{float(k.split('_')[1]):.0%}")
thr = section["thresholds"][key]
op = section["operating_points"][key]
st.sidebar.write(f"Test recall at this point: **{op['recall']:.3f}**")
st.sidebar.write(f"Measured test false-positive rate: **{op['false_positive_rate']:.3f}**")
st.sidebar.write(f"Estimated precision if attacks are 1% of traffic: **{op['est_precision_at_deploy_prevalence']:.3f}**")

st.subheader("Score flow records")
up = st.file_uploader("CSV with the same feature columns as training (optional)", type=["csv"])
if up:
    df = pd.read_csv(up, low_memory=False); df.columns = [str(c).strip() for c in df.columns]
else:
    sample = C.ARTIFACTS / "sample_test.csv"
    if not sample.exists():
        st.info("Upload a CSV to score."); st.stop()
    df = pd.read_csv(sample); st.info("Showing a held-out test sample saved by experiment.py. Upload your own CSV to score other flows.")

missing = [c for c in feats if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}"); st.stop()
X = df[feats].copy()
for c in bundle["num_cols"]:
    X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
for c in bundle["cat_cols"]:
    X[c] = X[c].astype(str)
Z = bundle["pre"].transform(X)
if use_rf:
    df["detection_score"] = bundle["rf"].predict_proba(Z)[:, 1]
else:
    df["detection_score"] = -bundle["iforest"].decision_function(Z)
df["flagged"] = df["detection_score"] > thr

a, b = st.columns(2)
a.metric("Flows scored", f"{len(df):,}"); b.metric("Flagged", f"{int(df['flagged'].sum()):,} ({df['flagged'].mean():.1%})")
color = "attack_type" if "attack_type" in df.columns else "flagged"
st.plotly_chart(px.histogram(df, x="detection_score", color=color, nbins=60, barmode="overlay", opacity=.6,
                             title=f"{model_choice} score distribution (dashed line = threshold)").add_vline(x=thr, line_dash="dash"))
if "attack_type" in df.columns:
    t = df.groupby("attack_type")["flagged"].agg(rows="size", flagged_share="mean").reset_index()
    st.dataframe(t.rename(columns={"flagged_share": "share flagged"}))
st.dataframe(df.sort_values("detection_score", ascending=False).head(50))

st.subheader("Saved experiment results")
fam_section = M["supervised_rf"] if use_rf else M
fam = fam_section["per_family_recall"][key]
st.dataframe(pd.DataFrame([{"attack family": f, "test rows": v["rows"], "recall": round(v["recall"], 3)} for f, v in fam.items()]))
if M["leave_one_family_out"]:
    st.dataframe(pd.DataFrame(M["leave_one_family_out"]).round(3))
st.caption("Offline analysis of labelled flow records from one testbed. Not a real-time or deployed system.")
