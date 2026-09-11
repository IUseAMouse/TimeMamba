"""
Where does the validation loss come from? (2026-09-11: val_loss 1.29, twice
TimeJEPA's, while the GIFT CRPS is on par; train_loss_step spikes at 9-27.)

Runs the validation loader through a checkpoint and reports the per-item
pinball distribution (normalized space, the loss the module logs), the worst
items with the amplitude of their quantiles, and the same loss at several
Delta scales.

    CUDA_VISIBLE_DEVICES=1 python scripts/probe_val_loss.py <checkpoint> [--batches 300] [--batch 32] [hydra overrides]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TimeJEPA" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation.loading import load_checkpoint  # noqa: E402
from timejepa.models.decoders.quantile_head import pinball_loss  # noqa: E402
from timessm.model import build_from_config  # noqa: E402
from train_ssm import build_datamodule  # noqa: E402


def item_pinball(q, target, levels):
    """[B] pinball per item (x2 convention of the module's loss)."""
    out = []
    for b in range(q.shape[0]):
        out.append(float(pinball_loss(q[b:b + 1], target[b:b + 1], levels)))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default="ssm_mini_v3")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--scales", default="0.25,1,4")
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. model.ssm.d_model=32")
    args = ap.parse_args()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=[f"data.batch_size={args.batch}",
                                                          "data.num_workers=2", *args.overrides])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_config(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    head = model.decoder.decoder
    dm = build_datamodule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()
    scales = [float(s) for s in args.scales.split(",")]

    losses = {s: [] for s in scales}
    worst = []                                  # (loss, batch, item, stats)
    n_batches = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.batches:
                break
            x = batch["context"].to(device)
            y = batch["target"].to(device)
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            if y.ndim == 2:
                y = y.unsqueeze(-1)
            for s in scales:
                w = None if s == 1.0 else torch.full((x.shape[0],), s, device=device)
                out = model.forecast(x, w=w)
                t = model.robust_scaler.transform(y)
                t = (t - model.revin.mean) / model.revin.std
                q = out["quantiles"]
                per = item_pinball(q, t, head.quantile_levels)
                losses[s].extend(per.tolist())
                if s == 1.0:
                    for b in np.argsort(per)[-3:]:
                        worst.append((float(per[b]), bi, int(b),
                                      float(q[b].abs().max()), float(t[b].abs().max()),
                                      float(x[b].abs().max()), float(y[b].abs().max()),
                                      float(model.revin.std[b].max()),
                                      float(model.robust_scaler.scale[b].max())))
            n_batches += 1

    print(f"{n_batches} batches x {args.batch} items, checkpoint {Path(args.checkpoint).name}")
    for s in scales:
        a = np.asarray(losses[s])
        print(f"w={s:<5g} mean {a.mean():.4f}  median {np.median(a):.4f}  p90 {np.percentile(a, 90):.3f}  "
              f"p99 {np.percentile(a, 99):.3f}  max {a.max():.2f}  "
              f"share of mean from top 1% items: {np.sort(a)[-max(1, len(a) // 100):].sum() / a.sum():.1%}")
    worst.sort(reverse=True)
    print("\nworst items at w=1 (loss | batch,item | max|q| max|target| in normalized space | "
          "max|ctx| max|y| raw | revin std | robust scale)")
    for r in worst[:15]:
        print(f"{r[0]:8.2f} | {r[1]:3d},{r[2]:2d} | {r[3]:8.2f} {r[4]:8.2f} | {r[5]:10.3g} {r[6]:10.3g} | "
              f"{r[7]:9.3g} {r[8]:9.3g}")


if __name__ == "__main__":
    main()
