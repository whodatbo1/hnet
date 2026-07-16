"""Merge the metric history of two wandb runs where one resumes the other.

Training logs with an explicit global step (wandb.log(..., step=step)), so the
resumed run's _step values live in the same step space as the original run.
The merge takes the original run's rows up to the resumption's first step
(the original may have trained past the checkpoint before being stopped),
then appends all rows of the resumed run, and logs everything into a new run.

Example:
    python scripts/analysis/merge_wandb_runs.py wvmobnka 1iillouq
"""

import argparse

import wandb


def fetch_history(run):
    """Full-fidelity history (run.history() subsamples to ~500 points)."""
    rows = [row for row in run.scan_history() if "_step" in row]
    rows.sort(key=lambda r: r["_step"])
    return rows


def clean(row):
    """Drop wandb-internal keys (_step, _runtime, _timestamp, ...) and Nones."""
    return {k: v for k, v in row.items() if not k.startswith("_") and v is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", help="run id of the original run")
    parser.add_argument("resumed", help="run id of the run resumed from a checkpoint")
    parser.add_argument("--path", default="marko-ivanovv/hnet", help="entity/project")
    parser.add_argument("--name", default=None,
                        help="name for the merged run (default: <original name>_merged)")
    parser.add_argument("--cutoff", type=int, default=None,
                        help="step at which to switch to the resumed run "
                             "(default: first logged step of the resumed run)")
    args = parser.parse_args()

    api = wandb.Api()
    run_a = api.run(f"{args.path}/{args.original}")
    run_b = api.run(f"{args.path}/{args.resumed}")

    print(f"Fetching history of {run_a.name} ({args.original})...")
    hist_a = fetch_history(run_a)
    print(f"  {len(hist_a)} rows, steps {hist_a[0]['_step']}..{hist_a[-1]['_step']}")
    print(f"Fetching history of {run_b.name} ({args.resumed})...")
    hist_b = fetch_history(run_b)
    print(f"  {len(hist_b)} rows, steps {hist_b[0]['_step']}..{hist_b[-1]['_step']}")

    cutoff = args.cutoff if args.cutoff is not None else hist_b[0]["_step"]
    kept_a = [r for r in hist_a if r["_step"] < cutoff]
    dropped = len(hist_a) - len(kept_a)
    print(f"Cutoff step {cutoff}: keeping {len(kept_a)} rows from the original run "
          f"(dropping {dropped} rows past the resumption point)")

    entity, project = args.path.split("/")
    name = args.name or f"{run_a.name}_merged"
    with wandb.init(entity=entity, project=project, name=name, config=dict(run_a.config),
                    job_type="merge", notes=f"Merge of {args.original} + {args.resumed}") as run:
        for row in kept_a + hist_b:
            payload = clean(row)
            if payload:
                run.log(payload, step=int(row["_step"]))

    print(f"Done: {run.url}")


if __name__ == "__main__":
    main()
