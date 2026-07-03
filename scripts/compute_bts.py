"""Compute the Bilingual Transfer Score (BTS) from two wandb runs.

Definition (ATLAS):
    BTS_{s->t} = -(sigma_bi(L_t(d_mono)) - 2 * d_mono) / d_mono

where d_mono is a pre-defined target data budget (tokens/bytes) for the
monolingual target-language run, L_t(d) is the monolingual model's val loss at
d, and sigma_bi(l) is the TOTAL number of tokens (both languages) the 50/50
bilingual model needs to reach val loss l on the target language.
BTS = 0 means no transfer (bilingual needs exactly 2*d_mono total tokens,
i.e. the same amount of target-language data), > 0 positive transfer,
< 0 interference.

Token counts are derived as step * tokens_per_step (inferred from the logged
total_tokens/step slope) rather than the raw total_tokens series, because
total_tokens resets to 0 when a run resumes from a checkpoint.

Usage:
    python scripts/compute_bts.py --mono-run 1s2zdct4 --bi-run wvmobnka
    python scripts/compute_bts.py ... --d-mono 6e9        # explicit budget
    python scripts/compute_bts.py ... --metric bpb        # use val/bpb curves
"""

import argparse

import numpy as np
import wandb


def fetch_series(run, keys):
    """Return columns for `keys` from all history rows that contain them all."""
    cols = {k: [] for k in keys}
    for row in run.scan_history(keys=keys):
        if any(row.get(k) is None for k in keys):
            continue
        for k in keys:
            cols[k].append(row[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def tokens_per_step(run):
    """Median of d(total_tokens)/d(step) over the train log; robust to the
    total_tokens reset on resume (negative/zero diffs are discarded)."""
    s = fetch_series(run, ["total_tokens", "step"])
    if len(s["step"]) >= 2:
        order = np.argsort(s["step"])
        dt = np.diff(s["total_tokens"][order])
        ds = np.diff(s["step"][order])
        ok = (dt > 0) & (ds > 0)
        if ok.any():
            return float(np.median(dt[ok] / ds[ok]))
    cfg = run.config.get("train", {})
    return float(cfg["batch_size"] * cfg["grad_accum_steps"] * cfg["seq_len"])


def val_curve(run, metric_key, tps):
    """(tokens, loss) arrays of the run's validation points, sorted, deduped
    (last value wins per step, e.g. re-validation after a resume)."""
    s = fetch_series(run, [metric_key, "step"])
    by_step = {}
    for step, v in zip(s["step"], s[metric_key]):
        by_step[step] = v
    steps = np.array(sorted(by_step))
    losses = np.array([by_step[st] for st in steps])
    return steps * tps, losses


def first_crossing(tokens, losses, target):
    """Total tokens at which `losses` first reaches `target` (linear
    interpolation between the bracketing validation points), or None."""
    below = np.nonzero(losses <= target)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return float(tokens[0])  # already at/below target at first val point
    t0, t1 = tokens[i - 1], tokens[i]
    l0, l1 = losses[i - 1], losses[i]
    frac = (l0 - target) / (l0 - l1) if l1 < l0 else 1.0
    return float(t0 + frac * (t1 - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono-run", required=True, help="wandb run id of the monolingual target run")
    ap.add_argument("--bi-run", required=True, help="wandb run id of the 50/50 bilingual run")
    ap.add_argument("--entity", default="marko-ivanovv")
    ap.add_argument("--project", default="hnet")
    ap.add_argument("--target", default="hplt-nld_Latn", help="target-language source name in the bilingual run")
    ap.add_argument("--metric", default="loss", choices=["loss", "bpb"], help="val metric used for matching")
    ap.add_argument("--d-mono", type=float, default=None,
                    help="target budget in tokens/bytes (default: every mono val point; final = largest)")
    args = ap.parse_args()

    api = wandb.Api()
    mono = api.run(f"{args.entity}/{args.project}/{args.mono_run}")
    bi = api.run(f"{args.entity}/{args.project}/{args.bi_run}")
    print(f"mono: {mono.name} ({mono.state})")
    print(f"bi:   {bi.name} ({bi.state})")

    tps_mono, tps_bi = tokens_per_step(mono), tokens_per_step(bi)
    mono_tok, mono_loss = val_curve(mono, f"val/{args.metric}", tps_mono)
    bi_tok, bi_loss = val_curve(bi, f"val/{args.metric}/{args.target}", tps_bi)
    # Target-language token counter of the bilingual run (for reporting)
    s = fetch_series(bi, [f"total_tokens/{args.target}", "step"])
    tgt_frac = np.median(s[f"total_tokens/{args.target}"] / (s["step"] * tps_bi)) if len(s["step"]) else 0.5

    print(f"mono: {len(mono_tok)} val points up to {mono_tok[-1]/1e9:.2f}B tokens "
          f"(final val/{args.metric} = {mono_loss[-1]:.4f})")
    print(f"bi:   {len(bi_tok)} val points up to {bi_tok[-1]/1e9:.2f}B total tokens "
          f"({100*tgt_frac:.1f}% {args.target}; final val/{args.metric}/{args.target} = {bi_loss[-1]:.4f})")
    print()

    if args.d_mono is not None:
        if not (mono_tok[0] <= args.d_mono <= mono_tok[-1]):
            raise SystemExit(f"--d-mono {args.d_mono:.3g} outside mono val range "
                             f"[{mono_tok[0]:.3g}, {mono_tok[-1]:.3g}]")
        d_grid = np.array([args.d_mono])
        l_grid = np.array([np.interp(args.d_mono, mono_tok, mono_loss)])
    else:
        d_grid, l_grid = mono_tok, mono_loss

    header = f"{'d_mono (B)':>10}  {'L_t(d_mono)':>11}  {'sigma_bi (B)':>12}  {'tgt toks (B)':>12}  {'BTS':>7}"
    print(header)
    print("-" * len(header))
    last = None
    for d, l in zip(d_grid, l_grid):
        sigma = first_crossing(bi_tok, bi_loss, l)
        if sigma is None:
            # Bilingual never reached l within its trained budget: bound only
            bound = -(bi_tok[-1] - 2 * d) / d
            print(f"{d/1e9:>10.3f}  {l:>11.4f}  {'not reached':>12}  {'':>12}  {'<' + format(bound, '.3f'):>7}")
            continue
        bts = -(sigma - 2 * d) / d
        print(f"{d/1e9:>10.3f}  {l:>11.4f}  {sigma/1e9:>12.3f}  {sigma*tgt_frac/1e9:>12.3f}  {bts:>7.3f}")
        last = (d, l, sigma, bts)

    print()
    if last is None:
        print("BTS not computable: the bilingual run never reached any mono val loss level.")
    else:
        d, l, sigma, bts = last
        print(f"BTS_{{en->nl}} at d_mono={d/1e9:.3f}B: {bts:.3f}")
        print(f"  (mono loss {l:.4f} reached by bilingual at {sigma/1e9:.3f}B total tokens; "
              f"~{sigma*tgt_frac/1e9:.3f}B {args.target} tokens vs {d/1e9:.3f}B monolingual)")
        if args.d_mono is None and last[0] < mono_tok[-1]:
            print(f"  NOTE: largest computable d_mono is {d/1e9:.3f}B; beyond that the bilingual run "
                  f"has not (yet) reached the mono loss — extend the bilingual run for larger budgets.")


if __name__ == "__main__":
    main()
