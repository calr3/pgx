"""Summarise gess_slide_diag.py output: network ranks of good captures by slide distance."""
import json, sys, collections
import numpy as np

recs = json.load(open(sys.argv[1]))
by_pos = collections.defaultdict(list)
for r in recs:
    by_pos[r["pos"]].append(r)
played_v = {p: next(r["value"] for r in rs if r["played"]) for p, rs in by_pos.items()}
for r in recs:
    r["gain"] = r["value"] - played_v[r["pos"]]
    r["capture"] = r["opp_cap"] > 0 or r["win"]

BINS = [(1, 1), (2, 3), (4, 6), (7, 17)]


def row(rs):
    if not rs:
        return "      -"
    rs_src = np.array([r["r_src"] for r in rs])
    rd = np.array([r["r_dst"] for r in rs])
    return (f"{len(rs):6d}  src rank med {np.median(rs_src):4.0f}  in top16 {np.mean(rs_src < 16):5.1%}  "
            f"p_src med {np.median([r['p_src'] for r in rs]):.3f}  | dst top1 {np.mean(rd == 0):5.1%}  "
            f"dst rank med {np.median(rd):3.0f}/{np.median([r['n_dst'] for r in rs]):.0f}  "
            f"played {np.mean([r['played'] for r in rs]):5.1%}")


def table(title, pred):
    print(f"\n{title}")
    for lo, hi in BINS:
        rs = [r for r in recs if r["capture"] and lo <= r["dist"] <= hi and pred(r)]
        print(f"  dist {lo:2d}-{hi:<2d} {row(rs)}")


print(f"{len(by_pos)} positions, {len(recs)} moves, {sum(r['capture'] for r in recs)} captures")
table("All captures (opponent stones taken, own ring kept)", lambda r: True)
table("Captures at least as good as the move played (gain >= 0)", lambda r: r["gain"] >= 0)
table("Captures clearly better than the move played (gain >= 0.2)", lambda r: r["gain"] >= 0.2)
table("Captures clearly better, taking 2+ stones or winning", lambda r: r["gain"] >= 0.2 and (r["opp_cap"] >= 2 or r["win"]))

# per position: was the best capture a long one, and did the played move match?
print("\nPositions whose best evaluated move is a capture clearly better than the one played:")
cnt = collections.Counter()
for p, rs in by_pos.items():
    best = max(rs, key=lambda r: r["value"])
    if best["capture"] and best["gain"] >= 0.2:
        cnt[next(f"{lo}-{hi}" for lo, hi in BINS if lo <= best["dist"] <= hi)] += 1
print("  by distance of that best capture:", dict(sorted(cnt.items())), "of", len(by_pos), "positions")
