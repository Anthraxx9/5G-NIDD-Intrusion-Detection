# 5G NIDD Intrusion Detection System

**Status: In Progress**

A machine learning project exploring network intrusion detection using flow-level traffic features from the 5G-NIDD dataset.

The project compares an unsupervised anomaly detection approach with a supervised classification model to investigate the detection of malicious network traffic and the challenges of identifying attack families not encountered during training.

## Objectives

- Explore and preprocess network-flow traffic data.
- Build an intrusion detection pipeline using flow-level features.
- Evaluate unsupervised anomaly detection using Isolation Forest.
- Compare its performance with a supervised Random Forest classifier.
- Investigate generalization to previously unseen attack families.
- Document evaluation results, limitations, and future improvements.

## Methods and Models

### Isolation Forest
An unsupervised anomaly detection model trained to identify observations that differ from the learned distribution of benign traffic.

### Random Forest
A supervised classification baseline used to distinguish benign and attack traffic based on labelled examples.

### Evaluation
The experiment pipeline examines ROC-AUC, PR-AUC, recall at selected false-positive rates, and leave-one-attack-family-out evaluation.

Reported results are preliminary and require further validation before conclusions about generalization or real-world deployment can be drawn.

## Project Structure

```text
5G-NIDD-Intrusion-Detection/
├── app.py
├── config.py
├── experiment.py
├── make_synthetic.py
├── nidd.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `app.py` — application entry point.
- `config.py` — project configuration.
- `experiment.py` — model experiments and evaluation.
- `make_synthetic.py` — synthetic data generation utilities.
- `nidd.py` — dataset loading, preprocessing, feature selection, and splitting.
- `requirements.txt` — Python dependencies.

## Dataset

The project uses the 5G-NIDD dataset, containing labelled network-flow observations.

The dataset is not included in this repository. Obtain it from its original source, review its licence and terms of use, and configure the local dataset path before running experiments.

The current pipeline uses flow-level traffic features. This should not be interpreted as validated detection of attacks within 5G-core signalling or production 5G infrastructure.

## Getting Started

### Requirements

- Python 3.10 or a compatible version supported by the dependencies
- Git
- The dataset configured locally

### Installation

Clone the repository:

```bash
git clone https://github.com/Anthraxx9/5G-NIDD-Intrusion-Detection.git
cd 5G-NIDD-Intrusion-Detection
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Running experiments

After configuring the dataset path in accordance with the project's configuration:

```bash
python experiment.py --split random --data "PATH_TO_DATASET.csv"
```

Replace `PATH_TO_DATASET.csv` with the actual local path to your dataset.

Review `config.py` and the command-line options in `experiment.py` if your local configuration differs.

## Current Work

- Dataset loading and feature preprocessing.
- Unsupervised and supervised model comparison.
- Evaluation across held-out attack families.
- Result validation and documentation.

## Limitations

- Results depend on the dataset, its labelling, and the evaluation split.
- Correlated or duplicated flows may affect measured performance.
- Random train/test splits may overestimate generalization when related observations appear in multiple splits.
- Strong supervised results require further investigation for possible leakage and dataset-specific shortcuts.
- Flow-level benchmark results do not establish real-world production performance.

## Future Improvements

- Audit preprocessing and split integrity for data leakage.
- Validate results using stronger group-based or time-based splits where appropriate.
- Investigate false positives and false negatives by attack family.
- Improve reproducibility and document experimental settings.
- Explore additional datasets and more realistic deployment conditions.

## Disclaimer

This is an educational and experimental machine learning project. It is not a production-ready security system, and its current results should not be treated as evidence of operational effectiveness.

## Author

**Raj Dwivedi**

- GitHub: [@Anthraxx9](https://github.com/Anthraxx9)
