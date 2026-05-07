import wandb
api = wandb.Api()

def get_run_by_name(path, name):
    runs = api.runs(path, filters={"displayName": name})
    runs = list(runs)
    if not runs:
        raise ValueError(f"No run named {name}")
    if len(runs) > 1:
        print(f"Warning: {len(runs)} runs match, using first")
    return runs[0]

run_a = get_run_by_name("marko-ivanovv/hnet", "train_hnet_1stage_XXS_entropy_10B_bytes_2026-17-04-14-41-16")
run_b = get_run_by_name("marko-ivanovv/hnet", "train_hnet_1stage_XXS_baseline_10B_bytes_fast_2026-17-04-15-14-39")

ha = run_a.history(keys=["_step", "loss"], pandas=True).set_index("_step")
hb = run_b.history(keys=["_step", "loss"], pandas=True).set_index("_step")

ha_bpb = run_a.history(keys=["_step", "val/bpb"], pandas=True).set_index("_step")
hb_bpb = run_b.history(keys=["_step", "val/bpb"], pandas=True).set_index("_step")

diff = (ha["loss"] - hb["loss"]).dropna()
diff_bpb = (ha_bpb["val/bpb"] - hb_bpb["val/bpb"]).dropna()

with wandb.init(project="hnet_analysis", name="diff_hnet_XXS_entropy_minus_baseline", job_type="analysis") as run:
    run.define_metric("loss_diff_step")
    run.define_metric("loss_diff", step_metric="loss_diff_step")
    run.define_metric("bpb_diff_step")
    run.define_metric("val/bpb_diff", step_metric="bpb_diff_step")

    for step, v in diff.items():
        run.log({"loss_diff": v, "loss_diff_step": int(step)})
    for step, v in diff_bpb.items():
        run.log({"val/bpb_diff": v, "bpb_diff_step": int(step)})