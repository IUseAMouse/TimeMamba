"""
Time one training step of the spike model on a random batch and list the
operators that pay for it (pod at 2.7-3 it/s on 2026-09-10, six times
TimeJEPA's per-sample cost; the per-step tokenization is the suspect).

    python scripts/profile_step.py                      # spike config, batch 128
    python scripts/profile_step.py --batch 64 --expand 1
    python scripts/profile_step.py --no-profile         # timing only
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm.model import build_from_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ssm_mini_v3")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--expand", type=int, default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config)
    if args.expand is not None:
        cfg.model.ssm.expand = args.expand
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_config(cfg).to(device).train()
    head = model.decoder.decoder
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"{model.count_parameters():,} parameters, device {device}, batch {args.batch}, "
          f"context {args.context}, horizon {cfg.model.prediction_length}")

    x = 50 + torch.randn(args.batch, args.context, 1, device=device)
    y = 50 + torch.randn(args.batch, cfg.model.prediction_length, 1, device=device)
    w = torch.full((args.batch,), 0.5, device=device)

    def step():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.forecast(x, w=w)
            target = model.robust_scaler.transform(y)
            target = (target - model.revin.mean) / model.revin.std
            loss = head.loss(out["quantiles"], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    print(f"{dt * 1e3:.1f} ms/step -> {1 / dt:.2f} it/s, {args.batch / dt:.0f} samples/s")
    if device.type == "cuda":
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.no_profile:
        return

    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(3):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=20))


if __name__ == "__main__":
    main()
