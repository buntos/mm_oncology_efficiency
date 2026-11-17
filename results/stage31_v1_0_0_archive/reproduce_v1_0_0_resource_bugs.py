#!/usr/bin/env python3
r"""Reproduce the two resource-accounting defects of Stage 31 v1.0.0 (CPU, synthetic tiny model; no data, no GPU).
  .\.venv\Scripts\python.exe -B results\stage31_v1_0_0_archive\reproduce_v1_0_0_resource_bugs.py
Loads the ARCHIVED v1.0.0 script copy in this folder (SHA256 checked). Writes reproduce_v1_0_0_resource_bugs.json here.
"""
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "31_train_patch_budget_baselines_v1.0.0.py"
assert hashlib.sha256(SRC.read_bytes()).hexdigest() == "776a66428592079bfe882a385ee09f10c7a3541a765438ebc01c6cb4b41e6b4d"
spec = importlib.util.spec_from_file_location("stage31_v100", SRC)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ROOT = HERE.parents[1]  # the archived copy lives in results\..., point it at the project as the original did
m.SCRIPTS = {k: m.ROOT / "scripts" / v.name for k, v in m.SCRIPTS.items()}
m.load_stage29(); m.init_torch(False)
s29, torch = m.s29, m.torch


def loader(pid, epoch=0):
    i = int(pid[-4:]); g = torch.Generator().manual_seed(1000 * epoch + i)
    x = torch.rand((64, 3, 32, 32), generator=g) + (i % 2) * .5  # enough work for measurable wall time
    return x, torch.tensor([i % 2])


def runner(d, ces, **kw):
    mod = s29.build_model(None, 20261113, device=torch.device("cpu"), tiny=True)
    o = s29.optimizer_for(mod); o.param_groups[0]["lr"] = 1e-2; o.param_groups[1]["lr"] = 1e-2
    tr = [f"TCGA-AA-{i:04d}" for i in range(1, 14)]
    labels = {p: ("LUAD" if int(p[-4:]) % 2 == 0 else "LUSC") for p in tr}
    val = lambda model, ep: ({"ce": ces[ep - 1], "auroc": .5, "balanced_accuracy": .5, "macro_f1": .5},
                             [{"epoch": ep, "case_submitter_id": "X", "label": 0, "p_LUSC": .5}])
    r = m.Runner(run_dir=d, identity={"t": 1}, model=mod, optimizer=o, scaler=s29.scaler_for(torch.device("cpu")),
                 device=torch.device("cpu"), train_ids=tr, labels=labels, seed=20261113, budget=16, max_epochs=len(ces),
                 patience=5, loader=loader, validate_fn=val, train_eval_fn=lambda model: {"ce": 0.0}, checkpoint_every=2, **kw)
    r.frozen_bn = s29.bn_state(mod)
    return r


out = {"script_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    # Defect 1: stop after training phase (before validation) loses training wall/peaks in the checkpoint.
    a = runner(t / "a", [.7, .6]); a.start()
    try:
        a.run()
    except m.StopRequested:
        pass
    b = runner(t / "b", [.7, .6], stop_at_phase=(1, "trained")); b.start()
    try:
        b.run()
    except m.StopRequested:
        pass
    in_memory = b.state["epoch_res"]["train_wall_seconds"]
    saved = b.checkpoints.load()[0]["progress"]["epoch_res"]["train_wall_seconds"]
    b2 = runner(t / "b", [.7, .6]); b2.start(); b2.run()
    final_row = b2.state["resources"][0]
    ca, cb = a.checkpoints.load()[0], b2.checkpoints.load()[0]
    out["defect1_stop_after_training_phase"] = {
        "in_memory_train_wall_seconds_at_stop": in_memory, "checkpointed_train_wall_seconds": saved,
        "final_epoch1_train_wall_seconds_after_resume": final_row["train_wall_seconds"],
        "final_epoch1_train_peak_rss_bytes_after_resume": final_row["train_sampled_peak_rss_bytes"],
        "continuous_epoch1_train_wall_seconds": a.state["resources"][0]["train_wall_seconds"],
        "weights_equal_to_continuous": all(s29.tree_equal(ca[k], cb[k]) for k in ("model", "optimizer", "scaler")),
        "reproduced": in_memory > 0 and saved == 0 and final_row["train_wall_seconds"] == 0}
    # Mid-epoch intermediate checkpoint: accumulated training wall of the current segment is not in the checkpoint.
    c = runner(t / "c", [.7, .6], stop_after_attempts=2); c.start()
    try:
        c.run()
    except m.StopRequested:
        pass
    out["defect1b_intermediate_checkpoint_mid_training"] = {
        "checkpointed_train_wall_seconds_after_2_attempts": c.checkpoints.load()[0]["progress"]["epoch_res"]["train_wall_seconds"],
        "reproduced": c.checkpoints.load()[0]["progress"]["epoch_res"]["train_wall_seconds"] == 0}
    # Defect 2: finished run - checkpoint cost/count added after serialisation; reopening shows smaller values.
    mem_row = a.state["resources"][-1]
    disk_row = a.checkpoints.load()[0]["progress"]["resources"][-1]
    out["defect2_commit_checkpoint_cost_after_serialisation"] = {
        "in_memory_last_epoch_checkpoint_count": mem_row["checkpoint_count"], "checkpointed_last_epoch_checkpoint_count": disk_row["checkpoint_count"],
        "in_memory_checkpoint_io_wall_seconds": mem_row["checkpoint_io_wall_seconds"], "checkpointed_checkpoint_io_wall_seconds": disk_row["checkpoint_io_wall_seconds"],
        "reproduced": disk_row["checkpoint_count"] < mem_row["checkpoint_count"]}
(HERE / "reproduce_v1_0_0_resource_bugs.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
sys.exit(0 if all(v["reproduced"] for k, v in out.items() if k.startswith("defect")) else 1)
