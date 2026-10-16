"""Loading, feature selection and splitting for the 5G-NIDD flow CSV (column names are detected, not assumed)."""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import config as C

LABEL_CANDIDATES = ["label", "binary label", "binary_label", "class", "is_attack"]
ATTACK_CANDIDATES = ["attack type", "attack_type", "attacktype", "attack", "attack_category",
                     "category", "attack label", "attack_label"]
TIME_CANDIDATES = ["starttime", "start_time", "start time", "timestamp", "stime", "time", "ts", "seq"]  # "seq" = last-resort ordering
ID_EXACT = {"seq", "starttime", "lasttime", "stime", "ltime", "timestamp", "time", "flowid", "id",
            "sport", "dport", "srcport", "dstport", "attacktool", "tool"}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _tokens(name):
    return [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def _find(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def is_id_like(name):
    """Identifier/time-type columns that would let a model memorise the testbed instead of the traffic."""
    n, toks = _norm(name), _tokens(name)
    return (n in ID_EXACT or n.startswith("unnamed") or "addr" in n
            or "ip" in toks or "port" in toks or "mac" in toks)


def _order_key(series):
    """Return a chronological ordering only for genuinely date-like values.

    Numeric sequence columns (especially Seq) are identifiers, not timestamps.
    pandas can parse integers as nanoseconds since epoch, which creates a false
    impression that a sequence column is a real timestamp.
    """
    if series is None or pd.api.types.is_numeric_dtype(series):
        return None
    sample = series.dropna().astype(str).head(1000)
    if sample.empty:
        return None
    # Require date-like strings rather than arbitrary categorical/string IDs.
    t = pd.to_datetime(series, errors="coerce")
    if t.notna().mean() < 0.9:
        return None
    return t.astype("int64").to_numpy()


def load(path=None):
    path = path or C.DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = C.LABEL_COL or _find(df, LABEL_CANDIDATES)
    attack_col = C.ATTACK_COL or _find(df, ATTACK_CANDIDATES)
    time_col = C.TIME_COL or _find(df, TIME_CANDIDATES)
    if label_col is None and attack_col is None:
        raise ValueError(f"No label column found. Set LABEL_COL / ATTACK_COL in config.py. Columns: {list(df.columns)}")

    benign = {str(v).lower() for v in C.BENIGN_VALUES}
    is_benign = lambda s: s.astype(str).str.strip().str.lower().isin(benign)

    src = label_col if label_col is not None else attack_col
    df["is_attack"] = (~is_benign(df[src])).astype(int)
    if attack_col is not None:
        fam = df[attack_col].astype(str).str.strip().to_numpy(dtype=object)
        fam = np.where(df["is_attack"].to_numpy() == 1, fam, "Benign")
        fam = np.where((df["is_attack"].to_numpy() == 1) & is_benign(df[attack_col]).to_numpy(), "Unlabelled attack", fam)
        df["attack_type"] = fam
    else:
        df["attack_type"] = np.where(df["is_attack"] == 1, "Attack", "Benign")
    return df, dict(label_col=label_col, attack_col=attack_col, time_col=time_col)


def select_features(df, cols):
    """Returns X, num_cols, cat_cols, dropped{col: reason}. Label/identifier/constant/label-duplicate columns are removed."""
    y = df["is_attack"].to_numpy()
    skip = {"is_attack", "attack_type"} | {c for c in (cols["label_col"], cols["attack_col"]) if c}
    num_cols, cat_cols, dropped = [], [], {}
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if c in C.EXTRA_DROP:
            dropped[c] = "EXTRA_DROP (config)"
        elif is_id_like(c) or c == cols["time_col"]:
            dropped[c] = "identifier / time column"
        elif s.nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            v = s.astype(float).fillna(-9).to_numpy()
            if np.array_equal(v, y) or np.array_equal(v, 1 - y):
                dropped[c] = "exact copy of the label (leak)"
            elif s.notna().all() and s.nunique() == len(s) and np.array_equal(v, np.round(v)):
                dropped[c] = "unique integer per row (row identifier)"
            else:
                num_cols.append(c)
        elif s.nunique(dropna=True) <= 20:
            cat_cols.append(c)
        else:
            dropped[c] = "high-cardinality text"
    X = df[num_cols + cat_cols].copy()
    X[num_cols] = X[num_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X, num_cols, cat_cols, dropped


def drop_duplicate_flows(X, df):
    keep = ~X.assign(_t=df["attack_type"].to_numpy()).duplicated().to_numpy()
    return keep


def split(attack_type, order=None, mode=None):
    """Returns (train_idx, val_idx, test_idx) as position arrays."""
    mode = mode or C.SPLIT_MODE
    attack_type = np.asarray(attack_type)
    idx = np.arange(len(attack_type))
    if mode == "random":
        counts = pd.Series(attack_type).value_counts()
        strat = attack_type if counts.min() >= 3 else None
        tr, rest = train_test_split(idx, test_size=1 - C.TRAIN_FRAC, stratify=strat, random_state=C.RANDOM_STATE)
        strat2 = attack_type[rest] if strat is not None and pd.Series(attack_type[rest]).value_counts().min() >= 2 else None
        va, te = train_test_split(rest, test_size=1 - C.VAL_FRAC / (1 - C.TRAIN_FRAC), stratify=strat2, random_state=C.RANDOM_STATE)
        return tr, va, te
    tr, va, te = [], [], []
    for fam in np.unique(attack_type):
        ids = idx[attack_type == fam]
        if order is not None:
            ids = ids[np.argsort(order[ids], kind="stable")]
        a, b = int(len(ids) * C.TRAIN_FRAC), int(len(ids) * (C.TRAIN_FRAC + C.VAL_FRAC))
        tr.append(ids[:a]); va.append(ids[a:b]); te.append(ids[b:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def single_feature_auc(X, num_cols, y, max_rows=50_000):
    """How well each raw numeric feature alone separates attack from benign (1.0 = perfect = suspicious)."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(len(y), min(max_rows, len(y)), replace=False)
    ys, out = y[idx], []
    if len(np.unique(ys)) < 2:
        return out
    for c in num_cols:
        v = X[c].iloc[idx].fillna(X[c].median()).to_numpy()
        if np.nanstd(v) == 0:
            continue
        a = roc_auc_score(ys, v)
        out.append((c, float(max(a, 1 - a))))
    return sorted(out, key=lambda t: -t[1])


def order_for(df, cols):
    return _order_key(df[cols["time_col"]]) if cols["time_col"] else None
