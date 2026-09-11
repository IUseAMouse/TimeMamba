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
    return out


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
    missing = sorted(set(rb) - set(ra))
    if missing:
        print(f"missing from A ({len(missing)}): " + ", ".join(missing))


if __name__ == "__main__":
    main()
