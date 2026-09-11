"""
Fair read of a PARTIAL eval: leaderboard aggregates of two eval directories
on the configs BOTH have (an eval cut by OOM keeps the easy configs and
drops the long ones, so its 55-config geomean is not the champion's
97-config one).

    python scripts/compare_subset.py <eval_dir_a> <eval_dir_b>
    python scripts/compare_subset.py \
        evaluation/timessm_mini_v3_zs/epoch00_valloss1.3022/gift_flip_ratein-mix-pool \
        ../TimeJEPA/evaluation/timejepa_lotsa_mini_v3_head8_zs/<champion>/gift_flip_ratein-mix-pool
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TimeJEPA" / "src"))
from timejepa.evaluation import gift  # noqa: E402


def load(d: Path) -> dict:
    out = {}
    for f in (d / "per_config").glob("*.json"):
        r = json.loads(f.read_text())
        out[r["config"]] = r["model"]
        bt = (r.get("ratein") or {}).get("backtest") or {}
        out[r["config"]]["K"] = bt.get("K")
        mix = (r.get("ratein") or {}).get("mix")
        if mix:
            out[r["config"]]["K"] = "mix " + " ".join(f"k{k}:{w:.2f}" for k, w in mix["weights"].items())
    return out


def per_config_table(ra: dict, rb: dict, common: list, n: int = 12) -> None:
    """Largest CRPS ratios A/B in both directions, with the k each side chose."""
    rows = []
    for c in common:
        a, b = ra[c]["CRPS"], rb[c]["CRPS"]
        if a > 0 and b > 0:
            rows.append((a / b, c, a, b, ra[c].get("K"), rb[c].get("K")))
    rows.sort()
    print(f"\n{'config':34s} {'A':>7s} {'B':>7s} {'A/B':>6s}   kA | kB")
    for r in rows[:n]:
        print(f"{r[1]:34s} {r[2]:7.3f} {r[3]:7.3f} {r[0]:6.2f}   {r[4]} | {r[5]}")
    print("...")
    for r in rows[-n:]:
        print(f"{r[1]:34s} {r[2]:7.3f} {r[3]:7.3f} {r[0]:6.2f}   {r[4]} | {r[5]}")


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    sn = gift.official_seasonal_naive()
    print(f"{len(ra)} configs in A, {len(rb)} in B, {len(common)} in common")
    for name, r in (("A", ra), ("B", rb)):
        full = gift.aggregate(r, sn)
        sub = gift.aggregate({c: r[c] for c in common}, sn)
        print(f"{name}: all {full['n_configs_CRPS']:>3} configs  MASE {full['geomean_MASE_ratio']:.4f}  "
              f"CRPS {full['geomean_CRPS_ratio']:.4f}   |   common {sub['n_configs_CRPS']:>3}  "
              f"MASE {sub['geomean_MASE_ratio']:.4f}  CRPS {sub['geomean_CRPS_ratio']:.4f}")
    per_config_table(ra, rb, common)
    missing = sorted(set(rb) - set(ra))
    if missing:
        print(f"missing from A ({len(missing)}): " + ", ".join(missing))


if __name__ == "__main__":
    main()
