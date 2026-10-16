"""
python experiment.py --inspect          # look at the data and the feature choices, train nothing
python experiment.py                    # full experiment
python experiment.py --skip-heldout     # skip the slower leave-one-family-out comparison
Options: --data PATH   --split blocked|random
"""
import argparse, json, time
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import config as C
import nidd


def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    return ColumnTransformer(parts)


def scores_of(iforest, Z):
    return -iforest.decision_function(Z)          # higher = more anomalous


def threshold_for_fpr(benign_scores, fpr):
    return float(np.quantile(benign_scores, 1 - fpr))


def op_point(y, s, thr):
    pred = s > thr
    pos, neg = (y == 1).sum(), (y == 0).sum()
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    recall, fpr = tp / pos, fp / neg
    prec = tp / (tp + fp) if tp + fp else float("nan")
    pi = C.DEPLOY_PREVALENCE
    den = recall * pi + fpr * (1 - pi)
    return dict(threshold=thr, recall=float(recall), false_positive_rate=float(fpr), precision_on_test_mix=float(prec),
                est_precision_at_deploy_prevalence=float(recall * pi / den) if den > 0 else float("nan"))


def per_family_recall(fam, s, thr):
    out = {}
    for f in sorted(set(fam) - {"Benign"}):
        m = fam == f
        if m.sum() >= C.MIN_FAMILY_TEST_ROWS:
            out[f] = dict(rows=int(m.sum()), recall=float((s[m] > thr).mean()))
    return out


def fit_supervised_rf(Ztr, ytr):
    """Fit a supervised attack-vs-benign Random Forest, capped for memory use."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = np.arange(len(ytr))
    if len(idx) > C.MAX_SUPERVISED_ROWS:
        # Stratified sampling so the cap does not accidentally distort class balance.
        chosen = []
        for cls in np.unique(ytr):
            cidx = idx[ytr == cls]
            quota = max(1, int(C.MAX_SUPERVISED_ROWS * len(cidx) / len(idx)))
            chosen.extend(rng.choice(cidx, min(len(cidx), quota), replace=False).tolist())
        idx = np.asarray(chosen, dtype=int)
        if len(idx) > C.MAX_SUPERVISED_ROWS:
            idx = rng.choice(idx, C.MAX_SUPERVISED_ROWS, replace=False)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                n_jobs=-1, class_weight="balanced_subsample",
                                random_state=C.RANDOM_STATE)
    return rf.fit(Ztr[idx], ytr[idx])


def leave_one_family_out(Ztr, ytr, ftr, Zva, yva, Zte, yte, fte, if_scores_te, if_thr, families):
    """Isolation Forest (benign-only) vs Random Forest (supervised) on attack families the RF never saw in training."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    ben_va, ben_te = yva == 0, yte == 0

    def fit_rf(mask):
        idx = np.flatnonzero(mask)
        if len(idx) > C.MAX_SUPERVISED_ROWS:
            idx = rng.choice(idx, C.MAX_SUPERVISED_ROWS, replace=False)
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=C.RANDOM_STATE)
        return rf.fit(Ztr[idx], ytr[idx])

    def rf_thr(rf):
        return float(np.quantile(rf.predict_proba(Zva[ben_va])[:, 1], 1 - C.TARGET_FPR))

    rf_all = fit_rf(np.ones(len(ytr), bool)); thr_all = rf_thr(rf_all)
    rows = []
    for fam in families:
        rf = fit_rf(ftr != fam); thr = rf_thr(rf)
        m = fte == fam
        p_held = rf.predict_proba(Zte[m])[:, 1]
        rows.append(dict(
            attack_family=fam, test_rows=int(m.sum()),
            iforest_recall=float((if_scores_te[m] > if_thr).mean()),
            rf_recall_family_unseen=float((p_held > thr).mean()),
            rf_recall_family_seen=float((rf_all.predict_proba(Zte[m])[:, 1] > thr_all).mean()),
            rf_benign_fpr_unseen_run=float((rf.predict_proba(Zte[ben_te])[:, 1] > thr).mean())))
    return rows


def inspect(df, cols, X, num_cols, cat_cols, dropped):
    print(f"\nRows: {len(df):,}   Columns: {df.shape[1]}")
    print("Detected -> label:", cols["label_col"], "| attack-family:", cols["attack_col"], "| time:", cols["time_col"])
    print(f"Attack share of rows: {df['is_attack'].mean():.1%}")
    print("\nAttack families:\n", df["attack_type"].value_counts().to_string())
    print(f"\nKept {len(num_cols)} numeric + {len(cat_cols)} categorical features:", num_cols + cat_cols)
    print("\nDropped columns:")
    for c, why in dropped.items():
        print(f"  {c:25s} {why}")
    top = nidd.single_feature_auc(X, num_cols, df["is_attack"].to_numpy())[:8]
    print("\nStrongest single features (AUC, 1.0 = perfect; >0.99 deserves suspicion):")
    for c, a in top:
        print(f"  {c:25s} {a:.4f}{'   <-- LEAK?' if a > C.LEAK_AUC else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--data"); ap.add_argument("--split", choices=["blocked", "random"])
    args = ap.parse_args()
    if args.split:
        C.SPLIT_MODE = args.split
    t0 = time.time()

    df, cols = nidd.load(args.data)
    X, num_cols, cat_cols, dropped = nidd.select_features(df, cols)
    keep = nidd.drop_duplicate_flows(X, df)
    n_dupes = int((~keep).sum())
    df, X = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    if args.inspect:
        inspect(df, cols, X, num_cols, cat_cols, dropped); print(f"\nRemoved {n_dupes:,} duplicate flows."); return

    y, fam = df["is_attack"].to_numpy(), df["attack_type"].to_numpy(dtype=object)
    order = nidd.order_for(df, cols)
    tr, va, te = nidd.split(fam, order)
    print(f"Split ({C.SPLIT_MODE}): train {len(tr):,} / val {len(va):,} / test {len(te):,}   attack share {y.mean():.1%}")

    pre = make_preprocessor(num_cols, cat_cols)
    benign_tr = tr[y[tr] == 0]
    Z_ben = pre.fit_transform(X.iloc[benign_tr])
    iforest = IsolationForest(n_estimators=200, random_state=C.RANDOM_STATE, n_jobs=-1).fit(Z_ben)
    Ztr, Zva, Zte = (pre.transform(X.iloc[i]) for i in (tr, va, te))
    s_va, s_te = scores_of(iforest, Zva), scores_of(iforest, Zte)
    y_va, y_te, f_te = y[va], y[te], fam[te]

    thresholds = {f: threshold_for_fpr(s_va[y_va == 0], f) for f in C.REPORT_FPRS}
    primary_thr = threshold_for_fpr(s_va[y_va == 0], C.TARGET_FPR)
    op = {f"fpr_{f}": op_point(y_te, s_te, thresholds[f]) for f in C.REPORT_FPRS}
    fam_recall = {f"fpr_{f}": per_family_recall(f_te, s_te, thresholds[f]) for f in C.REPORT_FPRS}
    roc_auc, pr_auc = float(roc_auc_score(y_te, s_te)), float(average_precision_score(y_te, s_te))
    prevalence = float(y_te.mean())
    leaks = [(c, a) for c, a in nidd.single_feature_auc(X, num_cols, y) if a > C.LEAK_AUC]

    print(f"\nTEST  ROC-AUC {roc_auc:.3f}   PR-AUC {pr_auc:.3f}   (PR-AUC of a coin flip = attack share = {prevalence:.3f})")
    for f in C.REPORT_FPRS:
        o = op[f"fpr_{f}"]
        print(f"  @FPR {f:>4.0%}: recall {o['recall']:.3f} | measured FPR {o['false_positive_rate']:.3f} | "
              f"precision(test mix) {o['precision_on_test_mix']:.3f} | est. precision if attacks were {C.DEPLOY_PREVALENCE:.0%} of traffic: "
              f"{o['est_precision_at_deploy_prevalence']:.3f}")
    print(f"  'Flag everything' baseline: recall 1.000, precision {prevalence:.3f}")
    for c, a in leaks:
        print(f"  WARNING: feature '{c}' alone gives AUC {a:.4f} - possible leak; check it or add it to EXTRA_DROP.")

    # Supervised baseline: learns from both benign and attack rows in training only.
    rf = fit_supervised_rf(Ztr, y[tr])
    p_va_rf = rf.predict_proba(Zva)[:, 1]
    p_te_rf = rf.predict_proba(Zte)[:, 1]
    rf_thresholds = {f"fpr_{f}": threshold_for_fpr(p_va_rf[y_va == 0], f) for f in C.REPORT_FPRS}
    rf_ops = {k: op_point(y_te, p_te_rf, thr) for k, thr in rf_thresholds.items()}
    rf_roc_auc = float(roc_auc_score(y_te, p_te_rf))
    rf_pr_auc = float(average_precision_score(y_te, p_te_rf))
    rf_family_recall = {k: per_family_recall(f_te, p_te_rf, thr) for k, thr in rf_thresholds.items()}
    print(f"\nSUPERVISED RANDOM FOREST TEST  ROC-AUC {rf_roc_auc:.3f}   PR-AUC {rf_pr_auc:.3f}")
    for f in C.REPORT_FPRS:
        o = rf_ops[f"fpr_{f}"]
        print(f"  @FPR {f:>4.0%}: recall {o['recall']:.3f} | measured FPR {o['false_positive_rate']:.3f} | "
              f"precision(test mix) {o['precision_on_test_mix']:.3f} | est. precision at {C.DEPLOY_PREVALENCE:.0%} prevalence "
              f"{o['est_precision_at_deploy_prevalence']:.3f}")

    lofo = []
    if not args.skip_heldout:
        families = [f for f, v in fam_recall[f"fpr_{C.TARGET_FPR}"].items()]
        print(f"\nLeave-one-family-out: Isolation Forest vs Random Forest at {C.TARGET_FPR:.0%} FPR ({len(families)} families)...")
        lofo = leave_one_family_out(Ztr, y[tr], fam[tr], Zva, y_va, Zte, y_te, f_te, s_te, primary_thr, families)
        print(pd.DataFrame(lofo).round(3).to_string(index=False))

    # artifacts
    A = C.ARTIFACTS
    joblib.dump(dict(pre=pre, iforest=iforest, rf=rf, num_cols=num_cols, cat_cols=cat_cols), A / "model.joblib")
    metrics = dict(rows=int(len(df)), duplicates_removed=n_dupes, split_mode=C.SPLIT_MODE, attack_share=float(y.mean()),
                   test_attack_share=prevalence, n_features=len(num_cols) + len(cat_cols), features=num_cols + cat_cols,
                   dropped_columns=dropped, detected_columns=cols, roc_auc=roc_auc, pr_auc=pr_auc,
                   thresholds={f"fpr_{f}": thresholds[f] for f in C.REPORT_FPRS}, target_fpr=C.TARGET_FPR,
                   operating_points=op, per_family_recall=fam_recall,
                   supervised_rf=dict(roc_auc=rf_roc_auc, pr_auc=rf_pr_auc, thresholds=rf_thresholds,
                                      operating_points=rf_ops, per_family_recall=rf_family_recall),
                   leave_one_family_out=lofo,
                   leak_warnings=[dict(feature=c, auc=a) for c, a in leaks], seconds=round(time.time() - t0, 1))
    (A / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sample = df.iloc[te].sample(min(3000, len(te)), random_state=C.RANDOM_STATE)
    sample[num_cols + cat_cols + ["attack_type", "is_attack"]].to_csv(A / "sample_test.csv", index=False)

    fpr_c, tpr_c, _ = roc_curve(y_te, s_te)
    plt.figure(figsize=(5, 4.5)); plt.plot(fpr_c, tpr_c, label=f"Isolation Forest (AUC {roc_auc:.3f})"); plt.plot([0, 1], [0, 1], "--", c="grey")
    for f in C.REPORT_FPRS:
        o = op[f"fpr_{f}"]; plt.scatter(o["false_positive_rate"], o["recall"], zorder=3)
    plt.xlabel("False-positive rate"); plt.ylabel("Recall"); plt.title("ROC on test set"); plt.legend(); plt.tight_layout()
    plt.savefig(A / "roc.png", dpi=150); plt.close()
    if lofo:
        d = pd.DataFrame(lofo).set_index("attack_family")[["iforest_recall", "rf_recall_family_unseen", "rf_recall_family_seen"]]
        d.plot.bar(figsize=(9, 4.5)); plt.ylabel(f"Recall @ ~{C.TARGET_FPR:.0%} FPR"); plt.title("Recall by attack family")
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(A / "per_family.png", dpi=150); plt.close()
    write_report(metrics)
    print(f"\nSaved model, metrics.json, report.md, plots to {A}  ({metrics['seconds']}s)")


def write_report(m):
    L = ["# Experiment report (auto-generated)", "",
         f"- Flows: {m['rows']:,} after removing {m['duplicates_removed']:,} duplicates; attack share {m['attack_share']:.1%}; "
         f"split mode: {m['split_mode']}; features used: {m['n_features']}",
         f"- Isolation Forest trained on benign training flows only. Thresholds set from benign validation flows (no attack labels used).",
         f"- Test ROC-AUC {m['roc_auc']:.3f}, PR-AUC {m['pr_auc']:.3f} (coin-flip PR-AUC = {m['test_attack_share']:.3f})", "",
         "## Operating points (test set)", "", "| Target FPR | Measured FPR | Recall | Precision (test mix) | Est. precision @ deployment prevalence |", "|---|---|---|---|---|"]
    for k, o in m["operating_points"].items():
        L.append(f"| {k.split('_')[1]} | {o['false_positive_rate']:.3f} | {o['recall']:.3f} | {o['precision_on_test_mix']:.3f} | {o['est_precision_at_deploy_prevalence']:.3f} |")
    L += ["", f"## Recall per attack family at {m['target_fpr']:.0%} FPR", "", "| Family | Test rows | Recall |", "|---|---|---|"]
    for f, v in m["per_family_recall"][f"fpr_{m['target_fpr']}"].items():
        L.append(f"| {f} | {v['rows']} | {v['recall']:.3f} |")
    if m.get("supervised_rf"):
        rfm = m["supervised_rf"]
        L += ["", "## Supervised Random Forest (attack-vs-benign)", "",
              f"- Trained on both benign and attack training rows (maximum {C.MAX_SUPERVISED_ROWS:,} sampled rows).",
              f"- Test ROC-AUC {rfm['roc_auc']:.3f}, PR-AUC {rfm['pr_auc']:.3f}.", "",
              "| Target FPR | Measured FPR | Recall | Precision (test mix) | Est. precision @ deployment prevalence |",
              "|---|---|---|---|---|"]
        for k, o in rfm["operating_points"].items():
            L.append(f"| {k.split('_')[1]} | {o['false_positive_rate']:.3f} | {o['recall']:.3f} | {o['precision_on_test_mix']:.3f} | {o['est_precision_at_deploy_prevalence']:.3f} |")
        L += ["", f"### Random Forest recall per family at {m['target_fpr']:.0%} target FPR", "",
              "| Family | Test rows | Recall |", "|---|---:|---:|"]
        for f, v in rfm["per_family_recall"][f"fpr_{m['target_fpr']}"] .items():
            L.append(f"| {f} | {v['rows']} | {v['recall']:.3f} |")
    if m["leave_one_family_out"]:
        L += ["", "## Unseen-family test: Isolation Forest vs Random Forest", "",
              "Random Forest is retrained once per family with that family removed from training; both models use a ~same FPR. "
              "'seen' = Random Forest trained on all families (upper reference).", "",
              "| Family | IForest | RF (family unseen) | RF (family seen) |", "|---|---|---|---|"]
        for r in m["leave_one_family_out"]:
            L.append(f"| {r['attack_family']} | {r['iforest_recall']:.3f} | {r['rf_recall_family_unseen']:.3f} | {r['rf_recall_family_seen']:.3f} |")
    if m["leak_warnings"]:
        L += ["", "## Warnings", ""] + [f"- Feature `{w['feature']}` alone reaches AUC {w['auc']:.4f}: check for leakage." for w in m["leak_warnings"]]
    L += ["", "## Limitations (keep these in your README)", "",
          "- Offline analysis of labelled flow records from one 5G testbed; not real-time, not a deployed system.",
          "- Flow features are ordinary network-flow statistics, not 5G-core signalling.",
          "- Attack types are DoS floods and port scans only; results say nothing about other attack classes.",
          "- Precision depends on how common attacks are; the test mix is not a real network. The deployment-prevalence column is an estimate, not a measurement.",
          "- Even with a blocked split, flows from one capture session are correlated, so scores may be optimistic."]
    (C.ARTIFACTS / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
