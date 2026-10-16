"""All settings in one place. Run `python experiment.py --inspect` first, then edit this file if needed."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "5G-NIDD-combined.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Column names. None = auto-detect (check what --inspect reports and set explicitly if it guessed wrong).
LABEL_COL = None      # binary label column (benign vs attack)
ATTACK_COL = None     # attack-family column (e.g. UDPFlood, SYNScan)
TIME_COL = None       # timestamp column, used only for ordering the split, never as a feature

BENIGN_VALUES = {"benign", "normal", "0", "0.0", "benign_traffic"}
EXTRA_DROP = []       # extra column names to exclude from the features

# "blocked": within each attack family, earliest 60% -> train, next 20% -> val, last 20% -> test.
#            Reduces leakage from near-identical neighbouring flows. Uses TIME_COL if found, else file order.
# "random":  stratified random split (leaks more; use only for comparison).
SPLIT_MODE = "blocked"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2

# Operating points are set from BENIGN validation flows only: threshold = score quantile giving this false-positive rate.
TARGET_FPR = 0.05
REPORT_FPRS = [0.01, 0.05, 0.10]
DEPLOY_PREVALENCE = 0.01   # used to estimate precision if only 1% of real traffic were attacks

MAX_SUPERVISED_ROWS = 150_000   # cap for the Random Forest baseline's training rows
MIN_FAMILY_TEST_ROWS = 30       # skip per-family stats for families smaller than this in the test split
LEAK_AUC = 0.99                 # warn if one raw feature alone separates attack/benign this well
