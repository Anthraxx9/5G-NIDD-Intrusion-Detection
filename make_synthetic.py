"""Creates a SYNTHETIC stand-in with 5G-NIDD-like columns, for testing the pipeline before you have the real file.
Do not report results from this file as real-data results."""
import argparse
import numpy as np, pandas as pd
from pathlib import Path

FAM = {  # share, protocol probs (tcp, udp, icmp), duration mu/sigma (log), pkts lambda, bytes/pkt mean
    "Benign": (0.35, (.7, .25, .05), (0.0, 1.2), 12, 700),
    "ICMPFlood": (0.08, (0, 0, 1), (-3, .5), 40, 90),
    "UDPFlood": (0.09, (0, 1, 0), (-2.5, .5), 60, 120),
    "SYNFlood": (0.09, (1, 0, 0), (-3, .6), 30, 60),
    "HTTPFlood": (0.08, (1, 0, 0), (-.5, .8), 25, 500),
    "SlowrateDoS": (0.07, (1, 0, 0), (0.6, 1.1), 9, 650),      # deliberately close to benign
    "SYNScan": (0.08, (1, 0, 0), (-4, .4), 1, 60),
    "TCPConnectScan": (0.08, (1, 0, 0), (-3, .5), 3, 70),
    "UDPScan": (0.08, (0, 1, 0), (-4, .4), 1, 50),
}


def make(n=60000, seed=0, leak=False):
    rng = np.random.default_rng(seed)
    parts = []
    for i, (fam, (share, pp, (mu, sg), lam, bpp)) in enumerate(FAM.items()):
        k = int(n * share)
        proto = rng.choice(["tcp", "udp", "icmp"], k, p=np.array(pp) / sum(pp))
        dur = rng.lognormal(mu, sg, k)
        pk = rng.poisson(lam, k) + 1
        src_pk = np.maximum(1, (pk * rng.uniform(.4, .9, k)).astype(int)); dst_pk = pk - src_pk
        bytes_ = (pk * rng.normal(bpp, bpp * .25, k)).clip(40)
        t = np.sort(rng.uniform(0, 9e4, k)) if fam == "Benign" else np.sort(rng.uniform(i * 1e4, i * 1e4 + 8e3, k))
        parts.append(pd.DataFrame({
            "StartTime": 1.7e9 + t, "SrcAddr": rng.choice([f"10.0.0.{j}" for j in range(2, 40)], k),
            "DstAddr": rng.choice([f"10.0.1.{j}" for j in range(2, 10)], k),
            "Sport": rng.integers(1024, 65535, k), "Dport": rng.choice([80, 443, 53, 22, 8080, 5060], k),
            "Proto": proto, "Dur": dur, "SrcPkts": src_pk, "DstPkts": dst_pk, "TotPkts": pk,
            "SrcBytes": bytes_ * .6, "DstBytes": bytes_ * .4, "TotBytes": bytes_, "Rate": pk / np.maximum(dur, 1e-3),
            "SrcRate": src_pk / np.maximum(dur, 1e-3), "DstRate": dst_pk / np.maximum(dur, 1e-3),
            "sTtl": rng.choice([64, 128, 63], k), "dTtl": rng.choice([64, 128, 0], k), "Mean": dur / pk,
            "Label": "Benign" if fam == "Benign" else "Malicious", "Attack Type": fam,
            "Attack Tool": "none" if fam == "Benign" else "tool_" + fam[:3]}))
    df = pd.concat(parts).sort_values("StartTime").reset_index(drop=True)
    df.insert(0, "Seq", np.arange(len(df)))
    if leak:
        att = (df["Label"] != "Benign").astype(int)
        df["binary"] = att                                   # exact copy -> auto-dropped
        df["Hint"] = att + rng.normal(0, .05, len(df))       # near copy -> must trigger a warning
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/5G-NIDD-combined.csv")
    ap.add_argument("--leak", action="store_true"); a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); d = make(leak=a.leak); d.to_csv(a.out, index=False); print(f"wrote {len(d):,} synthetic rows to {a.out}; attack share {(d.Label!='Benign').mean():.1%}")
