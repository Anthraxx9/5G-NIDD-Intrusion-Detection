# Unsupervised anomaly detection on 5G-NIDD flows (offline analysis)

Research question: a model trained only on benign 5G network flows flags traffic that looks different. How well does it detect
attacks, and how does it compare to a supervised classifier on attack families the classifier never saw in training?

## Method
- Data: 5G-NIDD (University of Oulu 5G testbed; DoS floods and port scans), flow-level CSV. CC BY 4.0, cite the dataset paper.
- Isolation Forest trained on benign training flows only.
- Supervised Random Forest trained on both benign and attack training rows as a comparison candidate.
- Operating thresholds set from benign validation flows at fixed false-positive rates (1%, 5%, 10%). No attack labels are used to tune it.
- Metrics for both models: ROC-AUC, PR-AUC, recall at fixed FPR, per-attack-family recall, and an estimated precision if attacks were rare in real traffic.
- Unseen-family test: a Random Forest is retrained once per attack family with that family removed, and compared with the Isolation Forest.
- Split: per-family blocked split (earliest flows train, latest test) to reduce leakage between near-identical neighbouring flows. Numeric sequence-like columns such as `Seq` are not treated as real timestamps; if no valid timestamp exists, blocked mode follows file order.

## Run
```bash
pip install -r requirements.txt
# put the real CSV at data/5G-NIDD-combined.csv  (or: python make_synthetic.py  for a SYNTHETIC test file)
python experiment.py --inspect      # check detected columns, dropped columns, leakage warnings
python experiment.py                # full experiment -> artifacts/ (metrics.json, report.md, plots)
streamlit run app.py
```
The dashboard lets you switch between Isolation Forest and the supervised Random Forest. Use the real dataset for reported results; synthetic data is only a pipeline smoke test.

If `--inspect` guesses a column wrong, set LABEL_COL / ATTACK_COL / TIME_COL / EXTRA_DROP in `config.py`.

## Results
Paste the tables from `artifacts/report.md` here after running on the real data. Do not report numbers from the synthetic file.

## Limitations
- Offline analysis of labelled flow records from one testbed. Not real-time, not deployed.
- Features are ordinary flow statistics, not 5G-core signalling.
- Only DoS floods and port scans are covered.
- Precision depends on attack prevalence; the test mix is not a real network.
- Flows within one capture are correlated, so scores may still be optimistic.
