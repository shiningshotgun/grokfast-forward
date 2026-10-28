"""Grokfast runner: model/data/filter wiring, arms, pilot execution, checkpointing."""

import argparse
import json
import math
import os
import resource
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from model import Decoder
from data import multiplication_mod_p_data
from grokfast import gradfilter_ema

P = 97
SEQ_LEN = 5
BATCH_SIZE = 512
LR = 1e-3
BETAS = (0.9, 0.98)
ALPHA = 0.98
LAMB = 2.0
MAX_UPDATES_PER_ARM = 100
TOTAL_SECONDS = 600
STUDY_CAP = 50000
STUDY_ARM_SECONDS = 1500
STUDY_TOTAL_SECONDS = 6000
STUDY_NAME = "approved_comparison_20261020"
UPSTREAM_COMMIT = "93b2248e351d4bb9e4dda8ac32a6999b8970404b"


def build_model(seed=0):
    # independent stream for model init; seed selects the run's paired seed
    torch.manual_seed(seed)
    return Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=P + 2, seq_len=SEQ_LEN
    )


def build_data(seed=0):
    eq_token = P
    op_token = P + 1
    data = multiplication_mod_p_data(P, eq_token, op_token)
    g_split = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.shape[1], generator=g_split)
    train_idx, valid_idx = perm.split(data.shape[1] // 2)
    train_data = data[:, train_idx].clone()
    valid_data = data[:, valid_idx].clone()
    return train_data, valid_data, g_split


def make_order_gen(seed=0):
    return torch.Generator().manual_seed(seed)


def make_optimizer(model, weight_decay):
    return torch.optim.Adam(
        model.parameters(), lr=LR, betas=BETAS, weight_decay=weight_decay
    )


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )


def score_batch(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


def evaluate(model, data, batch_size, deadline, clock=time.monotonic):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    complete = True
    full_set = True
    with torch.no_grad():
        for batch in torch.split(data, batch_size, dim=1):
            if clock() >= deadline:
                complete = False
                full_set = False
                break
            inp = batch
            logits = model(inp[:-1])
            loss = F.cross_entropy(logits[-1], inp[-1])
            total_loss += loss.item() * inp.shape[-1]
            total_acc += (logits[-1].argmax(-1) == inp[-1]).float().sum().item()
            n += inp.shape[-1]
    if n == 0:
        return {
            "loss": None,
            "acc": None,
            "n": 0,
            "complete": complete,
            "full_set": False,
            "overrun": clock() >= deadline,
        }
    return {
        "loss": total_loss / n,
        "acc": total_acc / n,
        "n": n,
        "complete": complete,
        "full_set": (n == data.shape[1]),
        "overrun": clock() >= deadline,
    }


def run_updates(
    model,
    optimizer,
    scheduler,
    grads,
    train_data,
    g_order,
    n_updates,
    device,
    filter_name,
    deadline,
    train_records,
    start_counter=0,
    resume_state=None,
    clock=time.monotonic,
):
    """Run up to n_updates optimizer updates; append to train_records.

    Supports mid-epoch resume via resume_state dict.
    Returns (i, grads, g_order, state).
    """
    if resume_state is None:
        epoch = 0
        perm = None
        cursor = 0
        i = start_counter
    else:
        epoch = resume_state.get("epoch", 0)
        perm = resume_state.get("perm", None)
        if perm is not None:
            perm = perm.clone()
        cursor = resume_state.get("cursor", 0)
        i = resume_state.get("i", start_counter)
        if resume_state.get("order_gen_state", None) is not None:
            g_order.set_state(resume_state["order_gen_state"])

    state = {
        "epoch": epoch,
        "perm": perm,
        "cursor": cursor,
        "i": i,
        "order_gen_state": g_order.get_state(),
    }

    while i < n_updates and clock() < deadline:
        if perm is None:
            perm = torch.randperm(train_data.shape[1], generator=g_order)
            cursor = 0
        batches = list(torch.split(train_data[:, perm], BATCH_SIZE, dim=1))
        advanced = False
        while cursor < len(batches) and i < n_updates and clock() < deadline:
            batch = batches[cursor]
            update_start = clock()
            model.train()
            inp = batch.to(device)
            optimizer.zero_grad()
            logits = model(inp[:-1])
            loss = F.cross_entropy(logits[-1], inp[-1])
            batch_acc = score_batch(logits[-1], inp[-1])
            batch_loss = loss.item()
            loss.backward()
            if filter_name == "ema":
                grads = gradfilter_ema(model, grads=grads, alpha=ALPHA, lamb=LAMB)
            optimizer.step()
            scheduler.step()
            i += 1
            update_time = clock() - update_start
            overrun = clock() >= deadline
            train_records.append(
                {
                    "step": i,
                    "loss": batch_loss,
                    "acc": batch_acc,
                    "update_time_s": update_time,
                    "overrun": overrun,
                }
            )
            cursor += 1
            advanced = True
        if cursor >= len(batches):
            epoch += 1
            perm = None
            cursor = 0
        state = {
            "epoch": epoch,
            "perm": None if perm is None else perm.clone(),
            "cursor": cursor,
            "i": i,
            "order_gen_state": g_order.get_state(),
        }
        if i >= n_updates or clock() >= deadline:
            break
        if not advanced:
            break
    if perm is not None:
        state["perm"] = perm.clone()
    state["order_gen_state"] = g_order.get_state()
    state["i"] = i
    state["epoch"] = epoch
    state["cursor"] = cursor
    return i, grads, g_order, state


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    grads,
    train_data,
    valid_data,
    g_order,
    g_split,
    counter,
    train_records,
    val_records,
    status,
    error=None,
    epoch=None,
    resume_state=None,
    weight_decay=None,
):
    ckpt = {
        "status": status,
        "updates_done": counter,
        "weight_decay": weight_decay,
        "model_state": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_state": None
        if grads is None
        else {k: v.detach().cpu().clone() for k, v in grads.items()},
        "train_data": train_data.detach().cpu().clone(),
        "valid_data": valid_data.detach().cpu().clone(),
        "order_gen_state": g_order.get_state(),
        "split_gen_state": g_split.get_state(),
        "rng_state": torch.get_rng_state(),
        "train_records": train_records,
        "val_records": val_records,
        "error": error,
        "epoch": epoch,
        "resume_state": resume_state,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    wd = ckpt.get("weight_decay", 0.0)
    if wd is None:
        wd = 0.0
    optimizer = make_optimizer(model, wd)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler = make_scheduler(optimizer)
    scheduler.load_state_dict(ckpt["scheduler_state"])
    # Restore optimizer LR from saved scheduler state (constructor step side-effect).
    if getattr(scheduler, "_last_lr", None):
        for pg, lr in zip(optimizer.param_groups, scheduler._last_lr):
            pg["lr"] = lr
    grads = None
    if ckpt["ema_state"] is not None:
        grads = {k: v.clone() for k, v in ckpt["ema_state"].items()}
    g_order = torch.Generator()
    g_order.set_state(ckpt["order_gen_state"])
    g_split = torch.Generator()
    g_split.set_state(ckpt["split_gen_state"])
    torch.set_rng_state(ckpt["rng_state"])
    train_data = ckpt["train_data"]
    valid_data = ckpt["valid_data"]
    resume_state = ckpt.get("resume_state", None)
    return (
        model,
        optimizer,
        scheduler,
        grads,
        train_data,
        valid_data,
        g_order,
        g_split,
        ckpt,
        resume_state,
    )


def run_arm(arm, deadline, out_dir, device, clock=time.monotonic, seed=0):
    if clock() >= deadline:
        return {
            "arm": arm["name"],
            "filter": arm["filter"],
            "weight_decay": arm["wd"],
            "status": "deadline_expired",
            "updates_done": 0,
            "train_records": [],
            "val_records": [],
            "error": None,
            "epoch": 0,
            "overrun": True,
            "checkpoint_saved": False,
        }
    model = build_model(seed).to(device)
    train_data, valid_data, g_split = build_data(seed)
    g_order = make_order_gen(seed)
    optimizer = make_optimizer(model, arm["wd"])
    scheduler = make_scheduler(optimizer)
    grads = None
    i = 0
    train_records = []
    val_records = []
    status = "running"
    error = None
    state = None

    try:
        if clock() < deadline:
            vr = evaluate(model, valid_data, BATCH_SIZE, deadline, clock=clock)
            val_records.append({"step": 0, **vr, "overrun": vr.get("overrun", False)})
        i, grads, g_order, state = run_updates(
            model,
            optimizer,
            scheduler,
            grads,
            train_data,
            g_order,
            MAX_UPDATES_PER_ARM,
            device,
            arm["filter"],
            deadline,
            train_records,
            start_counter=0,
            resume_state=None,
            clock=clock,
        )
        if i >= MAX_UPDATES_PER_ARM and clock() < deadline:
            vr = evaluate(model, valid_data, BATCH_SIZE, deadline, clock=clock)
            val_records.append(
                {"step": MAX_UPDATES_PER_ARM, **vr, "overrun": vr.get("overrun", False)}
            )
        if i >= MAX_UPDATES_PER_ARM:
            final_vr = val_records[-1] if len(val_records) > 1 else None
            if (
                final_vr is not None
                and final_vr.get("complete")
                and final_vr.get("full_set")
            ):
                status = "completed_100_updates"
            else:
                status = "completed_100_updates_incomplete_validation"
        elif clock() >= deadline:
            status = "deadline_expired"
        else:
            status = "stopped"
    except Exception as e:
        status = "error"
        error = repr(e)

    checkpoint_saved = False
    if clock() < deadline and status != "deadline_expired":
        ckpt_path = os.path.join(out_dir, f"checkpoint_{arm['name']}.pt")
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            grads,
            train_data,
            valid_data,
            g_order,
            g_split,
            i,
            train_records,
            val_records,
            status,
            error=error,
            epoch=(state["epoch"] if state else 0),
            resume_state=state,
            weight_decay=arm["wd"],
        )
        checkpoint_saved = True

    return {
        "arm": arm["name"],
        "filter": arm["filter"],
        "weight_decay": arm["wd"],
        "alpha": ALPHA if arm["filter"] == "ema" else None,
        "lamb": LAMB if arm["filter"] == "ema" else None,
        "status": status,
        "updates_done": i,
        "train_records": train_records,
        "val_records": val_records,
        "error": error,
        "epoch": (state["epoch"] if state else 0),
        "checkpoint_saved": checkpoint_saved,
    }


def run_pilot(out_dir, clock=time.monotonic, seed=0):
    # guard: if records exist, don't run
    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir)]
        if existing:
            print("Existing pilot records found; not running pilot.")
            return None
    os.makedirs(out_dir, exist_ok=True)

    deadline_start = clock()
    deadline = deadline_start + TOTAL_SECONDS

    arms = [
        {"name": "none_wd0", "filter": "none", "wd": 0.0},
        {"name": "ema_wd0", "filter": "ema", "wd": 0.0},
        {"name": "none_wd0.005", "filter": "none", "wd": 0.005},
        {"name": "ema_wd0.005", "filter": "ema", "wd": 0.005},
    ]

    device = torch.device("cpu")
    torch.set_num_threads(4)

    results = []
    for arm in arms:
        if clock() >= deadline:
            r = {
                "arm": arm["name"],
                "filter": arm["filter"],
                "weight_decay": arm["wd"],
                "status": "not_run_deadline",
                "updates_done": 0,
                "train_records": [],
                "val_records": [],
                "error": None,
                "epoch": 0,
            }
            results.append(r)
            # save a stub arm file
            with open(os.path.join(out_dir, f"arm_{arm['name']}.json"), "w") as f:
                json.dump(r, f, indent=2)
            continue
        r = run_arm(arm, deadline, out_dir, device, clock=clock, seed=seed)
        results.append(r)
        with open(os.path.join(out_dir, f"arm_{arm['name']}.json"), "w") as f:
            json.dump(r, f, indent=2)

    work_end = clock()
    total_elapsed = work_end - deadline_start
    reporting_start = clock()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = ru.ru_maxrss

    # config
    config = {
        "source": {
            "repo": "ironjr/grokfast",
            "commit": UPSTREAM_COMMIT,
            "files": ["main.py", "grokfast.py", "README.md"],
            "urls": [
                "https://raw.githubusercontent.com/ironjr/grokfast/93b2248e351d4bb9e4dda8ac32a6999b8970404b/main.py",
                "https://raw.githubusercontent.com/ironjr/grokfast/93b2248e351d4bb9e4dda8ac32a6999b8970404b/grokfast.py",
                "https://raw.githubusercontent.com/ironjr/grokfast/93b2248e351d4bb9e4dda8ac32a6999b8970404b/README.md",
            ],
        },
        "task": {
            "p": P,
            "x": "0..96",
            "y": "1..96",
            "n_examples": 9312,
            "split": "50/50",
            "tokens": 5,
            "result_only": True,
            "vocab": 99,
        },
        "model": {"layers": 2, "width": 128, "heads": 4},
        "optimizer": {"name": "Adam", "lr": LR, "betas": list(BETAS)},
        "batch_size": BATCH_SIZE,
        "warmup": "LambdaLR lambda update: 1 if update > 10 else update/10 (10 updates)",
        "arms": arms,
        "ema": {"alpha": ALPHA, "lamb": LAMB},
        "limits": {
            "updates_per_arm": MAX_UPDATES_PER_ARM,
            "total_seconds": TOTAL_SECONDS,
        },
        "seeds": {
            "model": 0,
            "split": 0,
            "order": 0,
            "independent_streams": True,
            "note": "Model init uses torch.manual_seed(0); split and order use "
            "separate generators seeded 0. This differs from upstream's "
            "single global stream, so exact upstream trajectories are not "
            "reproduced; arms share identical init/split/order.",
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    try:
        import numpy as np

        numpy_ver = np.__version__
    except Exception:
        numpy_ver = None
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": numpy_ver,
        "backend": "cpu",
        "mps_available": torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "maxrss": maxrss,
        "maxrss_unit": "bytes (macOS ru_maxrss; Linux would be KB)",
    }
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    summary = {
        "run_id": os.path.basename(out_dir),
        "deadline_start_monotonic": deadline_start,
        "work_elapsed_s": total_elapsed,
        "reporting_overhead_s": clock() - reporting_start,
        "reporting_sample_endpoint": "after config/environment writes; excludes summary JSON and markdown report writes",
        "deadline_s": TOTAL_SECONDS,
        "maxrss_bytes": maxrss,
        "arms": [
            {
                "name": r["arm"],
                "status": r["status"],
                "updates_done": r["updates_done"],
                "val_at_0": r["val_records"][0] if len(r["val_records"]) > 0 else None,
                "val_at_100": r["val_records"][-1]
                if len(r["val_records"]) > 1
                else None,
                "train_last": r["train_records"][-1] if r["train_records"] else None,
            }
            for r in results
        ],
        "memory_limitations": (
            "ru_maxrss peak resident set of this process; on macOS reported in "
            "bytes. Includes interpreter/OS overhead; not per-arm; CPU only."
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # markdown report
    lines = []
    lines.append("# Grokfast resource pilot")
    lines.append("")
    lines.append(f"Run id: {os.path.basename(out_dir)}")
    lines.append(f"Total elapsed: {total_elapsed:.3f}s / {TOTAL_SECONDS}s")
    lines.append(f"Max RSS: {maxrss} bytes")
    lines.append("")
    lines.append("| Arm | Status | Updates | Val@0 acc | Val@100 acc |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        v0 = r["val_records"][0]["acc"] if r["val_records"] else None
        v100 = r["val_records"][-1]["acc"] if len(r["val_records"]) > 1 else None
        lines.append(
            f"| {r['arm']} | {r['status']} | {r['updates_done']} | {v0} | {v100} |"
        )
    lines.append("")
    with open(os.path.join(out_dir, "pilot_report.md"), "w") as f:
        f.write("\n".join(lines))

    return results, total_elapsed, maxrss


def run_study_arm(arm, cap, eval_every, checkpoint_every, deadline, out_dir, device,
                 clock, global_start, seed=0):
    model = build_model(seed).to(device)
    train_data, valid_data, g_split = build_data(seed)
    g_order = make_order_gen(seed)
    optimizer = make_optimizer(model, arm["wd"])
    scheduler = make_scheduler(optimizer)
    grads = None
    i = 0
    train_records = []
    val_records = []
    status = "running"
    error = None
    state = None
    last_eval_step = -1

    os.makedirs(out_dir, exist_ok=True)

    def flush():
        payload = {
            "arm": arm["name"],
            "filter": arm["filter"],
            "weight_decay": arm["wd"],
            "alpha": ALPHA if arm["filter"] == "ema" else None,
            "lamb": LAMB if arm["filter"] == "ema" else None,
            "status": status,
            "updates_done": i,
            "cap": cap,
            "eval_every": eval_every,
            "checkpoint_every": checkpoint_every,
            "train_records": train_records,
            "val_records": val_records,
            "error": error,
            "epoch": (state["epoch"] if state else 0),
            "last_eval_step": last_eval_step,
        }
        with open(os.path.join(out_dir, "arm_{}.json".format(arm["name"])), "w") as f:
            json.dump(payload, f, indent=2)

    try:
        if clock() < deadline:
            eval_start = clock()
            vr = evaluate(model, valid_data, BATCH_SIZE, deadline, clock=clock)
            eval_dur = clock() - eval_start
            val_records.append(
                {
                    "step": 0,
                    **vr,
                    "eval_duration_s": eval_dur,
                    "work_elapsed_s": clock() - global_start,
                    "overrun": vr.get("overrun", False),
                }
            )
            last_eval_step = 0
            flush()

        while i < cap and clock() < deadline:
            next_target = min(i + eval_every, cap)
            i, grads, g_order, state = run_updates(
                model,
                optimizer,
                scheduler,
                grads,
                train_data,
                g_order,
                next_target,
                device,
                arm["filter"],
                deadline,
                train_records,
                start_counter=i,
                resume_state=state,
                clock=clock,
            )

            if i >= next_target and clock() < deadline and last_eval_step != i:
                eval_start = clock()
                vr = evaluate(model, valid_data, BATCH_SIZE, deadline, clock=clock)
                eval_dur = clock() - eval_start
                val_records.append(
                    {
                        "step": i,
                        **vr,
                        "eval_duration_s": eval_dur,
                        "work_elapsed_s": clock() - global_start,
                        "overrun": vr.get("overrun", False),
                    }
                )
                last_eval_step = i
                flush()

            if (
                checkpoint_every
                and i > 0
                and i % checkpoint_every == 0
                and clock() < deadline
            ):
                ckpt_path = os.path.join(out_dir, "checkpoint_{}.pt".format(arm["name"]))
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    grads,
                    train_data,
                    valid_data,
                    g_order,
                    g_split,
                    i,
                    train_records,
                    val_records,
                    "in_progress",
                    error=None,
                    epoch=(state["epoch"] if state else 0),
                    resume_state=state,
                    weight_decay=arm["wd"],
                )

            if clock() >= deadline:
                break

        if clock() < deadline and (not val_records or val_records[-1]["step"] != i):
            eval_start = clock()
            vr = evaluate(model, valid_data, BATCH_SIZE, deadline, clock=clock)
            eval_dur = clock() - eval_start
            val_records.append(
                {
                    "step": i,
                    **vr,
                    "eval_duration_s": eval_dur,
                    "work_elapsed_s": clock() - global_start,
                    "overrun": vr.get("overrun", False),
                }
            )
            last_eval_step = i
            flush()

        if i >= cap:
            final_vr = val_records[-1] if val_records else None
            if (
                final_vr is not None
                and final_vr.get("complete")
                and final_vr.get("full_set")
            ):
                status = "completed_cap"
            else:
                status = "completed_cap_incomplete_final_validation"
        elif clock() >= deadline:
            status = "deadline_stopped"
        else:
            status = "stopped"
    except Exception as e:
        status = "error"
        error = repr(e)

    checkpoint_saved = False
    if clock() < deadline and status not in ("deadline_stopped", "error"):
        ckpt_path = os.path.join(out_dir, "checkpoint_{}.pt".format(arm["name"]))
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            grads,
            train_data,
            valid_data,
            g_order,
            g_split,
            i,
            train_records,
            val_records,
            status,
            error=error,
            epoch=(state["epoch"] if state else 0),
            resume_state=state,
            weight_decay=arm["wd"],
        )
        checkpoint_saved = True

    flush()
    return {
        "arm": arm["name"],
        "filter": arm["filter"],
        "weight_decay": arm["wd"],
        "alpha": ALPHA if arm["filter"] == "ema" else None,
        "lamb": LAMB if arm["filter"] == "ema" else None,
        "status": status,
        "updates_done": i,
        "cap": cap,
        "eval_every": eval_every,
        "checkpoint_every": checkpoint_every,
        "train_records": train_records,
        "val_records": val_records,
        "error": error,
        "epoch": (state["epoch"] if state else 0),
        "checkpoint_saved": checkpoint_saved,
        "last_eval_step": last_eval_step,
    }


def run_study(base_dir=None, clock=time.monotonic, seed=0, study_name=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if study_name is None:
        study_name = STUDY_NAME
    out_root = os.path.join(base_dir, "study_outputs")
    marker = os.path.join(out_root, study_name + ".started")
    results_dir = os.path.join(out_root, study_name)

    if os.path.exists(marker) or os.path.exists(results_dir):
        print("Refusing study: marker or results already exist.")
        return None

    os.makedirs(out_root, exist_ok=True)
    with open(marker, "w") as f:
        f.write("started_monotonic={}\n".format(clock()))
    os.makedirs(results_dir)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    arms = [
        {"name": "none_wd0", "filter": "none", "wd": 0.0},
        {"name": "ema_wd0", "filter": "ema", "wd": 0.0},
        {"name": "none_wd0.005", "filter": "none", "wd": 0.005},
        {"name": "ema_wd0.005", "filter": "ema", "wd": 0.005},
    ]

    global_start = clock()
    global_deadline = global_start + STUDY_TOTAL_SECONDS

    import hashlib

    def sha256_file(path):
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    config = {
        "source": {
            "repo": "ironjr/grokfast",
            "commit": UPSTREAM_COMMIT,
            "files": ["main.py", "grokfast.py", "README.md"],
        },
        "task": {
            "p": P,
            "x": "0..96",
            "y": "1..96",
            "n_examples": 9312,
            "split": "50/50",
            "tokens": 5,
            "result_only": True,
            "vocab": 99,
        },
        "model": {"layers": 2, "width": 128, "heads": 4},
        "optimizer": {"name": "Adam", "lr": LR, "betas": list(BETAS)},
        "batch_size": BATCH_SIZE,
        "warmup": "LambdaLR lambda update: 1 if update > 10 else update/10 (10 updates)",
        "arms": arms,
        "ema": {"alpha": ALPHA, "lamb": LAMB},
        "limits": {
            "updates_per_arm": STUDY_CAP,
            "per_arm_seconds": STUDY_ARM_SECONDS,
            "total_seconds": STUDY_TOTAL_SECONDS,
        },
        "seeds": {
            "model": seed,
            "split": seed,
            "order": seed,
            "independent_streams": True,
        },
        "study_name": study_name,
        "fingerprints": {
            "model.py": sha256_file(os.path.join(base_dir, "model.py")),
            "data.py": sha256_file(os.path.join(base_dir, "data.py")),
            "grokfast.py": sha256_file(os.path.join(base_dir, "grokfast.py")),
            "runner.py": sha256_file(os.path.join(base_dir, "runner.py")),
        },
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    try:
        import numpy as np

        numpy_ver = np.__version__
    except Exception:
        numpy_ver = None
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": numpy_ver,
        "backend": "cpu",
        "mps_available": torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "maxrss_unit": "bytes (macOS ru_maxrss)",
    }
    with open(os.path.join(results_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    results = []
    for arm in arms:
        arm_start = clock()
        arm_deadline = min(global_deadline, arm_start + STUDY_ARM_SECONDS)
        if clock() >= arm_deadline or clock() >= global_deadline:
            r = {
                "arm": arm["name"],
                "filter": arm["filter"],
                "weight_decay": arm["wd"],
                "status": "not_run_deadline",
                "updates_done": 0,
                "cap": STUDY_CAP,
                "eval_every": 100,
                "checkpoint_every": 5000,
                "train_records": [],
                "val_records": [],
                "error": None,
                "epoch": 0,
                "checkpoint_saved": False,
                "last_eval_step": None,
            }
            results.append(r)
            with open(os.path.join(results_dir, "arm_{}.json".format(arm["name"])), "w") as f:
                json.dump(r, f, indent=2)
            continue
        r = run_study_arm(
            arm,
            STUDY_CAP,
            100,
            5000,
            arm_deadline,
            results_dir,
            device,
            clock,
            global_start,
            seed=seed,
        )
        results.append(r)

    work_end = clock()
    work_elapsed = work_end - global_start
    reporting_start = clock()

    def first_crossing(val_records, threshold=0.95):
        for v in val_records:
            if v.get("full_set") and v.get("acc") is not None and v["acc"] >= threshold:
                return v["step"], v.get("work_elapsed_s")
        return None, None

    def last_complete(val_records):
        full = [v for v in val_records if v.get("full_set")]
        return full[-1] if full else None

    def best_complete(val_records):
        full = [v for v in val_records if v.get("full_set")]
        if not full:
            return None
        return max(full, key=lambda v: v.get("acc", float("-inf")))

    arm_summary = []
    for r in results:
        cross_step, cross_elapsed = first_crossing(r.get("val_records", []))
        lc = last_complete(r.get("val_records", []))
        bc = best_complete(r.get("val_records", []))
        off_grid = None
        if r.get("val_records"):
            last = r["val_records"][-1]
            if last.get("step", 0) % 100 != 0:
                off_grid = {"step": last["step"], "acc": last.get("acc"), "n": last.get("n")}
        arm_summary.append(
            {
                "name": r["arm"],
                "status": r["status"],
                "updates_done": r["updates_done"],
                "first_crossing_step": cross_step,
                "first_crossing_work_elapsed_s": cross_elapsed,
                "last_complete_step": lc["step"] if lc else None,
                "last_complete_acc": lc["acc"] if lc else None,
                "last_complete_loss": lc["loss"] if lc else None,
                "best_complete_step": bc["step"] if bc else None,
                "best_complete_acc": bc["acc"] if bc else None,
                "best_complete_loss": bc["loss"] if bc else None,
                "off_grid_final": off_grid,
                "censored": cross_step is None,
            }
        )

    pairs = [("none_wd0", "ema_wd0"), ("none_wd0.005", "ema_wd0.005")]
    speedups = []
    for base_name, ema_name in pairs:
        b = next((a for a in arm_summary if a["name"] == base_name), None)
        e = next((a for a in arm_summary if a["name"] == ema_name), None)
        if (
            b is not None
            and e is not None
            and b["first_crossing_step"] is not None
            and e["first_crossing_step"] is not None
        ):
            speedups.append(
                {
                    "pair": "{} vs {}".format(base_name, ema_name),
                    "update_ratio_baseline_over_ema": b["first_crossing_step"]
                    / e["first_crossing_step"],
                    "work_time_ratio_baseline_over_ema": b["first_crossing_work_elapsed_s"]
                    / e["first_crossing_work_elapsed_s"],
                }
            )
        else:
            speedups.append(
                {
                    "pair": "{} vs {}".format(base_name, ema_name),
                    "estimable": False,
                    "reason": "at least one arm did not reach 0.95 full validation",
                }
            )

    summary = {
        "run_id": STUDY_NAME,
        "global_deadline_start_monotonic": global_start,
        "work_elapsed_s": work_elapsed,
        "reporting_overhead_s": clock() - reporting_start,
        "reporting_sample_endpoint": "after arm writes and before summary/report writes; excludes final markdown write",
        "deadline_s": STUDY_TOTAL_SECONDS,
        "per_arm_deadline_s": STUDY_ARM_SECONDS,
        "cap_per_arm": STUDY_CAP,
        "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "arms": arm_summary,
        "crossing_speedups": speedups,
        "notes": [
            "One paired seed; not universal or multi-seed evidence.",
            "Pilot extrapolation ~61 minutes training across four 50k arms did not isolate validation overhead.",
            "100-minute ceiling promises neither completion nor generalization.",
        ],
    }
    with open(os.path.join(results_dir, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("# Approved capped comparison")
    lines.append("")
    lines.append("Work elapsed: {:.3f}s / {}s".format(work_elapsed, STUDY_TOTAL_SECONDS))
    lines.append("Max RSS: {} bytes".format(summary["maxrss_bytes"]))
    lines.append("")
    lines.append("| Arm | Status | Updates | First >=0.95 step | Last complete acc | Best complete acc |")
    lines.append("|---|---|---|---|---|---|")
    for a in arm_summary:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                a["name"],
                a["status"],
                a["updates_done"],
                a["first_crossing_step"],
                a["last_complete_acc"],
                a["best_complete_acc"],
            )
        )
    lines.append("")
    lines.append("Non-crossers are censored at the last complete observation.")
    lines.append("Off-grid final observations are listed in study_summary.json.")
    lines.append("")
    with open(os.path.join(results_dir, "study_report.md"), "w") as f:
        f.write("\n".join(lines))

    return results, work_elapsed, summary["maxrss_bytes"], clock() - reporting_start



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="run the resource pilot")
    ap.add_argument("--study", action="store_true", help="run the approved capped comparison")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=0,
                    help="paired seed for model init, split, and ordering (default 0)")
    ap.add_argument("--study-name", default=None,
                    help="study output name under study_outputs (default: approved comparison)")
    ap.add_argument("--inspect", action="store_true", help="inspect records")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    po = os.path.join(base, "pilot_outputs")

    if args.inspect:
        if os.path.exists(po):
            for root, dirs, files in os.walk(po):
                for fn in files:
                    print(os.path.join(root, fn))
        else:
            print("no pilot_outputs")
        return

    if args.pilot:
        # guard existing records
        if os.path.exists(po) and any(os.scandir(po)):
            print("ERROR: existing pilot records found in", po)
            print("Not running pilot. Inspect and decide manually.")
            sys.exit(2)
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = args.out or os.path.join(po, run_id)
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        results, elapsed, maxrss = run_pilot(out_dir, seed=args.seed)
        print("Pilot done. elapsed", elapsed, "maxrss", maxrss)
        for r in results:
            print(r["arm"], r["status"], r["updates_done"])
        return

    if args.study:
        base = os.path.dirname(os.path.abspath(__file__))
        out_dir = args.out or base
        result = run_study(out_dir, seed=args.seed, study_name=args.study_name)
        if result is None:
            print("Study refused or already present.")
            sys.exit(2)
        results, elapsed, maxrss, reporting = result
        print("Study done. elapsed", elapsed, "maxrss", maxrss, "reporting", reporting)
        for r in results:
            print(r["arm"], r["status"], r["updates_done"])
        return

    print("Use --pilot or --study, --inspect to list records.")


if __name__ == "__main__":
    main()
