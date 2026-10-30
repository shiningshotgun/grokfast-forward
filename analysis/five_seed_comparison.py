#!/usr/bin/env python3
"""five_seed_comparison.py - Comprehensive comparison of Grokfast-EMA vs Baseline across 5 seeds.

Implements the prospective analysis specification in docs/analysis_spec.md exactly:
- Primary metric: steps to sustained >=0.99 crossing (never drops below 0.99 through step 50000).
  Arms not crossing are treated as RIGHT-CENSORED at >50000.
- Paired differences: EMA minus none.
  - Crossing steps: computed ONLY for both-cross pairs, labelled as conditional subset.
    Sign convention: NEGATIVE means EMA crossed sooner.
  - Final validation accuracy: at step 50000.
    Sign convention: POSITIVE means EMA ended higher. Units: percentage points (pp).
- Marginal medians: computed across all 5 seeds with censored seeds ordered above observed crossings.
  Reported only when >= 3 crossings observed AND the 3rd ordered value is an observed crossing.
- Part B: Prefix-consistency check between superseded and canonical runs over shared updates.
"""

import csv
import json
import os
import statistics
import subprocess
import sys

# Matplotlib configuration (non-interactive backend)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "outputs")

SEEDS = [
    ("multiseed_seed1_attempt3", 1),
    ("multiseed_seed2", 2),
    ("multiseed_seed3", 3),
    ("multiseed_seed4_attempt2", 4),
    ("multiseed_seed5", 5),
]

ARMS = ["none_wd0", "ema_wd0", "none_wd0.005", "ema_wd0.005"]


def compute_marginal_median(observed_crossings, total_seeds=5):
    """Computes marginal median across total_seeds under right-censoring beyond the horizon.
    
    Censored seeds are ordered above any observed crossing.
    For total_seeds=5 (or any odd total_seeds), the median is at index total_seeds // 2.
    The median is estimable if and only if at least (total_seeds // 2 + 1) crossings were observed
    AND the median-index ordered value is an observed crossing.
    Otherwise returns None.
    """
    median_idx = total_seeds // 2
    if len(observed_crossings) > median_idx:
        sorted_crossings = sorted(observed_crossings)
        return sorted_crossings[median_idx]
    return None


def format_marginal_median(observed_crossings, total_seeds=5):
    """Formats marginal median string for reports."""
    m = compute_marginal_median(observed_crossings, total_seeds=total_seeds)
    if m is not None:
        return f"{m:,}"
    else:
        return f"not estimable within the 50,000-update horizon ({len(observed_crossings)}/{total_seeds} crossed)"


def classify_pair_and_diff(m_none, m_ema):
    """Classifies a paired comparison and computes crossing difference if uncensored.
    
    Returns (quadrant, diff_cross).
    diff_cross is an integer if both crossed, or None if either arm was censored.
    """
    if m_none["crossed"] and m_ema["crossed"]:
        quadrant = "both-cross"
        diff_cross = m_ema["cross_step"] - m_none["cross_step"]  # negative = EMA sooner
    elif m_ema["crossed"] and not m_none["crossed"]:
        quadrant = "EMA-only-cross"
        diff_cross = None
    elif m_none["crossed"] and not m_ema["crossed"]:
        quadrant = "none-only-cross"
        diff_cross = None
    else:
        quadrant = "neither"
        diff_cross = None
    return quadrant, diff_cross


def validate_and_load_records():
    """Validates inputs strictly per docs/analysis_spec.md and returns structured data."""
    print("Validating input records...")
    data = {}
    for dir_name, seed_num in SEEDS:
        dir_path = os.path.join(REPO_ROOT, "study_outputs", dir_name)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Missing expected directory: {dir_path}")

        # Config validation
        cfg_path = os.path.join(dir_path, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for s_key in ["model", "split", "order"]:
            if cfg["seeds"][s_key] != seed_num:
                raise ValueError(f"{dir_name} config seed mismatch: {s_key}={cfg['seeds'][s_key]} != {seed_num}")

        data[seed_num] = {}
        for arm_name in ARMS:
            arm_path = os.path.join(dir_path, f"arm_{arm_name}.json")
            if not os.path.isfile(arm_path):
                raise FileNotFoundError(f"Missing arm record: {arm_path}")
            with open(arm_path, "r", encoding="utf-8") as f:
                r = json.load(f)

            if r.get("status") != "completed_cap":
                raise ValueError(f"{dir_name} arm {arm_name} status != completed_cap: {r.get('status')}")
            if r.get("updates_done") != 50000:
                raise ValueError(f"{dir_name} arm {arm_name} updates_done != 50000: {r.get('updates_done')}")

            val_records = r.get("val_records", [])
            if len(val_records) != 501:
                raise ValueError(f"{dir_name} arm {arm_name} val_records count != 501: {len(val_records)}")
            if val_records[0]["step"] != 0 or val_records[-1]["step"] != 50000:
                raise ValueError(f"{dir_name} arm {arm_name} step range mismatch: [0, 50000] expected")

            # Check 100-update grid uniformity
            for i in range(len(val_records) - 1):
                if val_records[i + 1]["step"] - val_records[i]["step"] != 100:
                    raise ValueError(f"{dir_name} arm {arm_name} grid nonuniform at index {i}")

            data[seed_num][arm_name] = r

    print("All 5 seeds, 20 arm records successfully validated against spec requirements.")
    return data


def run_part_b_prefix_check():
    """Performs Part B prefix-consistency check against superseded attempts."""
    print("\n--- Part B: Prefix-Consistency Check ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs_to_check = [
        {
            "label": "Seed 4: none_wd0 (Attempt 1 vs Attempt 2)",
            "superseded_dir": "study_outputs/multiseed_seed4",
            "superseded_arm": "arm_none_wd0.json",
            "canonical_dir": "study_outputs/multiseed_seed4_attempt2",
            "canonical_arm": "arm_none_wd0.json",
            "seed": 4,
            "arm": "none_wd0",
        },
        {
            "label": "Seed 1: ema_wd0 (Attempt 2 vs Attempt 3)",
            "superseded_dir": "study_outputs/multiseed_seed1_attempt2",
            "superseded_arm": "arm_ema_wd0.json",
            "canonical_dir": "study_outputs/multiseed_seed1_attempt3",
            "canonical_arm": "arm_ema_wd0.json",
            "seed": 1,
            "arm": "ema_wd0",
        },
    ]

    csv_rows = []
    summary_lines = [
        "# Part B: Prefix-Consistency Check Report",
        "",
        "Empirical examination of the recorded validation-accuracy curves between initial and amended-budget reruns.",
        "Over the shared update range, recorded validation-accuracy curves at the shared evaluation points are compared.",
        "",
        "**Scope Note**: This comparison covers exactly two arm-pairs: Seed 4 `none_wd0` (Attempt 1 vs Attempt 2) and Seed 1 `ema_wd0` (Attempt 2 vs Attempt 3). The other rerun arms were not part of this prefix comparison, and no claim is made about whether superseded records exist for them. Exact agreement of the recorded validation-accuracy curves over the shared prefix for these two arm-pairs supports (cannot prove) treating those two reruns as the same trajectory. For the other rerun arms, comparability rests on same-code/same-seed/same-cap reasoning, stated as an assumption.",
        "",
    ]

    for p in pairs_to_check:
        # Load canonical record from disk
        canon_path = os.path.join(REPO_ROOT, p["canonical_dir"], p["canonical_arm"])
        with open(canon_path, "r", encoding="utf-8") as f:
            canon_data = json.load(f)

        # Load superseded record from archive branch (preserved git object) or disk
        super_path_disk = os.path.join(REPO_ROOT, p["superseded_dir"], p["superseded_arm"])
        if os.path.isfile(super_path_disk):
            with open(super_path_disk, "r", encoding="utf-8") as f:
                super_data = json.load(f)
            source_note = "on disk"
        else:
            rel_path = f"{p['superseded_dir']}/{p['superseded_arm']}"
            cmd = ["git", "show", f"archive/superseded-attempts:{rel_path}"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT)
            if res.returncode == 0:
                super_data = json.loads(res.stdout)
                source_note = "archive/superseded-attempts branch"
            else:
                super_data = None
                source_note = "unreadable / missing"

        if super_data is None:
            summary_lines.append(f"## {p['label']}\n- Status: Missing / Unreadable in {p['superseded_dir']}\n")
            continue

        super_vals = {v["step"]: v for v in super_data.get("val_records", [])}
        canon_vals = {v["step"]: v for v in canon_data.get("val_records", [])}
        shared_steps = sorted(set(super_vals.keys()) & set(canon_vals.keys()))

        diffs = []
        for s in shared_steps:
            acc_super = super_vals[s]["acc"]
            acc_canon = canon_vals[s]["acc"]
            diff = abs(acc_canon - acc_super)
            diffs.append(diff)
            csv_rows.append({
                "pair": p["label"],
                "seed": p["seed"],
                "arm": p["arm"],
                "step": s,
                "superseded_acc": acc_super,
                "canonical_acc": acc_canon,
                "abs_diff": diff,
            })

        max_diff = max(diffs) if diffs else 0.0
        mean_diff = (sum(diffs) / len(diffs)) if diffs else 0.0
        max_step = max(shared_steps) if shared_steps else 0

        summary_lines.append(f"### {p['label']}")
        summary_lines.append(f"- **Record Source**: {source_note}")
        summary_lines.append(f"- **Shared Maximum Update**: {max_step:,}")
        summary_lines.append(f"- **Aligned Evaluation Count**: {len(shared_steps)}")
        summary_lines.append(f"- **Maximum Absolute Accuracy Difference**: {max_diff:.10f}")
        summary_lines.append(f"- **Mean Absolute Accuracy Difference**: {mean_diff:.10f}")
        summary_lines.append(f"- **Curve Agreement**: {'Exact match of recorded validation accuracy across all shared evaluation points (max diff 0.0000000000)' if max_diff == 0.0 else f'Divergence observed (max {max_diff:.6f})'}")
        summary_lines.append("")

        print(f"{p['label']}: {len(shared_steps)} aligned steps (up to {max_step}), max diff={max_diff:.10f}, mean diff={mean_diff:.10f}")

    # Write CSV
    csv_file = os.path.join(OUTPUT_DIR, "prefix_check.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair", "seed", "arm", "step", "superseded_acc", "canonical_acc", "abs_diff"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write MD
    md_file = os.path.join(OUTPUT_DIR, "prefix_check.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Prefix check CSV written to {csv_file}")
    print(f"Prefix check MD written to {md_file}")


def compute_crossing_step(val_records, threshold=0.99):
    """Calculates step of first evaluation >= threshold that NEVER drops below threshold through final step.
    
    Returns (crossing_step, True) if crossed; (None, False) if right-censored.
    """
    candidates = [v for v in val_records if v.get("acc") is not None and v["acc"] >= threshold]
    for c in candidates:
        c_step = c["step"]
        subsequent = [v for v in val_records if v["step"] >= c_step]
        if all(v.get("acc") is not None and v["acc"] >= threshold for v in subsequent):
            return c_step, True
    return None, False


def run_part_c_analysis(data):
    """Runs the 5-seed comparative analysis implementing docs/analysis_spec.md."""
    print("\n--- Part C: Five-Seed Comparative Analysis ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Per-seed per-arm table
    per_arm_rows = []
    arm_metrics = {}  # (seed, arm): {"cross_step": ..., "crossed": ..., "final_acc": ...}

    for seed_num in sorted(data.keys()):
        for arm_name in ARMS:
            r = data[seed_num][arm_name]
            val_records = r["val_records"]
            cross_step, crossed = compute_crossing_step(val_records, threshold=0.99)
            final_acc = val_records[-1]["acc"]

            arm_metrics[(seed_num, arm_name)] = {
                "cross_step": cross_step,
                "crossed": crossed,
                "final_acc": final_acc,
            }

            per_arm_rows.append({
                "seed": seed_num,
                "arm": arm_name,
                "filter": r["filter"],
                "weight_decay": r["weight_decay"],
                "crossing_update": cross_step if crossed else "RIGHT-CENSORED",
                "right_censored": not crossed,
                "final_val_accuracy": final_acc,
            })

    # Write per-seed arms CSV
    per_arm_csv = os.path.join(OUTPUT_DIR, "per_seed_arms.csv")
    with open(per_arm_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "arm", "filter", "weight_decay", "crossing_update", "right_censored", "final_val_accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_arm_rows)

    # 2. Paired analysis per seed x wd
    wd_levels = [0.0, 0.005]
    paired_rows = []
    paired_summary = {}

    for wd in wd_levels:
        wd_str = "0" if wd == 0.0 else "0.005"
        none_name = f"none_wd{wd_str}"
        ema_name = f"ema_wd{wd_str}"

        both_cross_diffs = []
        both_cross_faster_ema_count = 0
        final_acc_diffs_pp = []
        quadrant_counts = {"both-cross": 0, "EMA-only-cross": 0, "none-only-cross": 0, "neither": 0}
        faster_ema_count = 0
        higher_ema_count = 0

        for seed_num in sorted(data.keys()):
            m_none = arm_metrics[(seed_num, none_name)]
            m_ema = arm_metrics[(seed_num, ema_name)]

            # Quadrant classification & difference
            quadrant, diff_cross = classify_pair_and_diff(m_none, m_ema)
            quadrant_counts[quadrant] += 1

            if quadrant == "both-cross":
                both_cross_diffs.append(diff_cross)
                if diff_cross < 0:
                    both_cross_faster_ema_count += 1
                    faster_ema_count += 1
            elif quadrant == "EMA-only-cross":
                faster_ema_count += 1  # EMA crossed while none censored -> EMA faster

            diff_acc_pp = (m_ema["final_acc"] - m_none["final_acc"]) * 100.0  # percentage points (pp)
            final_acc_diffs_pp.append(diff_acc_pp)
            if diff_acc_pp > 0:
                higher_ema_count += 1

            paired_rows.append({
                "weight_decay": wd,
                "seed": seed_num,
                "none_arm": none_name,
                "none_crossing": m_none["cross_step"] if m_none["crossed"] else "RIGHT-CENSORED",
                "ema_arm": ema_name,
                "ema_crossing": m_ema["cross_step"] if m_ema["crossed"] else "RIGHT-CENSORED",
                "pair_quadrant": quadrant,
                "crossing_diff_ema_minus_none": diff_cross if diff_cross is not None else "N/A (censored)",
                "none_final_acc": m_none["final_acc"],
                "ema_final_acc": m_ema["final_acc"],
                "final_acc_diff_ema_minus_none_pp": diff_acc_pp,
            })

        # Marginal crossings for none and ema
        none_crossings = [arm_metrics[(s, none_name)]["cross_step"] for s in sorted(data.keys()) if arm_metrics[(s, none_name)]["crossed"]]
        ema_crossings = [arm_metrics[(s, ema_name)]["cross_step"] for s in sorted(data.keys()) if arm_metrics[(s, ema_name)]["crossed"]]

        paired_summary[wd] = {
            "wd": wd,
            "quadrant_counts": quadrant_counts,
            "both_cross_count": len(both_cross_diffs),
            "both_cross_diffs": both_cross_diffs,
            "both_cross_median": statistics.median(both_cross_diffs) if both_cross_diffs else "N/A (0 pairs)",
            "both_cross_range": (min(both_cross_diffs), max(both_cross_diffs)) if both_cross_diffs else "N/A",
            "both_cross_faster_ema_count": both_cross_faster_ema_count,
            "none_cross_count": len(none_crossings),
            "none_marginal_median": format_marginal_median(none_crossings, total_seeds=5),
            "ema_cross_count": len(ema_crossings),
            "ema_marginal_median": format_marginal_median(ema_crossings, total_seeds=5),
            "final_acc_diffs_pp": final_acc_diffs_pp,
            "final_acc_diff_median_pp": statistics.median(final_acc_diffs_pp),
            "final_acc_diff_range_pp": (min(final_acc_diffs_pp), max(final_acc_diffs_pp)),
            "faster_ema_count": faster_ema_count,
            "higher_ema_count": higher_ema_count,
        }

    # Write paired CSV
    paired_csv = os.path.join(OUTPUT_DIR, "paired_comparison.csv")
    with open(paired_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "weight_decay", "seed", "none_arm", "none_crossing", "ema_arm", "ema_crossing",
            "pair_quadrant", "crossing_diff_ema_minus_none", "none_final_acc", "ema_final_acc",
            "final_acc_diff_ema_minus_none_pp"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    # 3. Summary Markdown Report
    summary_md_file = os.path.join(OUTPUT_DIR, "five_seed_summary.md")
    md_lines = [
        "# Five-Seed Comparative Analysis: Grokfast-EMA vs Baseline",
        "",
        "## 1. Study Configuration & Prospective Methodology",
        "- **Evaluation Grid**: 501 evaluation steps on [0, 50,000] updates (every 100 updates).",
        "- **Primary Crossing Threshold**: First step with sustained validation accuracy $\\ge 0.99$ (never falling below 0.99 through step 50,000).",
        "- **Censoring Rule**: Non-crossing arms are strictly right-censored at $> 50,000$ updates.",
        "- **Sign Convention**: $\\Delta_{\\text{cross}} = \\text{EMA} - \\text{none}$ (negative favors EMA); $\\Delta_{\\text{acc}} = \\text{EMA} - \\text{none}$ (positive favors EMA).",
        "- **Accuracy Difference Units**: Reported as percentage points (pp).",
        "- **Budget Disclosure & Trajectory Comparability**: Seeds 1 (attempt 3) and 4 (attempt 2) were run under amended 2,100s/arm budgets (total 9,000s); Seeds 2, 3, 5 under 1,500s/arm budgets (total 6,000s). All 5 seeds executed under update cap 50,000 with identical random seed streams and hyperparameters. Exact agreement of recorded validation-accuracy curves over the shared prefix for the two reported arm-pairs supports (cannot prove) treating those two reruns as the same trajectory. For the other rerun arms, comparability rests on same-code/same-seed/same-cap reasoning, stated as an assumption.",
        "",
        "## 2. Per-Seed / Per-Arm Results Table",
        "",
        "| Seed | Arm | Filter | Weight Decay | Crossing Step ($\ge 0.99$ sustained) | Final Val Acc (Step 50k) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in per_arm_rows:
        cross_display = f"{r['crossing_update']:,}" if isinstance(r['crossing_update'], int) else "**RIGHT-CENSORED**"
        md_lines.append(f"| {r['seed']} | `{r['arm']}` | {r['filter']} | {r['weight_decay']} | {cross_display} | {r['final_val_accuracy'] * 100:.2f}% |")

    md_lines.extend([
        "",
        "## 3. Paired Comparisons within Weight Decay Levels",
        "",
        "| Weight Decay | Seed | Classification | Baseline Crossing | EMA Crossing | $\\Delta_{\\text{cross}}$ (EMA - none) | Baseline Final Acc | EMA Final Acc | $\\Delta_{\\text{acc}}$ (EMA - none) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in paired_rows:
        c_none = f"{r['none_crossing']:,}" if isinstance(r['none_crossing'], int) else "CENSORED"
        c_ema = f"{r['ema_crossing']:,}" if isinstance(r['ema_crossing'], int) else "CENSORED"
        d_cross = f"{r['crossing_diff_ema_minus_none']:,}" if isinstance(r['crossing_diff_ema_minus_none'], (int, float)) else r['crossing_diff_ema_minus_none']
        md_lines.append(
            f"| {r['weight_decay']} | {r['seed']} | `{r['pair_quadrant']}` | {c_none} | {c_ema} | {d_cross} | "
            f"{r['none_final_acc']*100:.2f}% | {r['ema_final_acc']*100:.2f}% | {r['final_acc_diff_ema_minus_none_pp']:+.2f} pp |"
        )

    md_lines.extend([
        "",
        "## 4. Aggregate Comparison & Sign Consistency",
        "",
    ])

    for wd in wd_levels:
        ps = paired_summary[wd]
        md_lines.append(f"### Weight Decay = {wd}")
        md_lines.append(f"- **Pair Quadrants**: `both-cross` = {ps['quadrant_counts']['both-cross']}/5, `EMA-only-cross` = {ps['quadrant_counts']['EMA-only-cross']}/5, `none-only-cross` = {ps['quadrant_counts']['none-only-cross']}/5, `neither` = {ps['quadrant_counts']['neither']}/5.")
        md_lines.append(f"- **Marginal Crossing Median (Baseline)**: {ps['none_marginal_median']}")
        md_lines.append(f"- **Marginal Crossing Median (EMA)**: {ps['ema_marginal_median']}")
        if ps['both_cross_count'] > 0:
            md_lines.append(f"- **Conditional Paired Crossing Difference** (conditional subset $n = {ps['both_cross_count']}/5$ pairs): median = {ps['both_cross_median']:,} updates, range = [{ps['both_cross_range'][0]:,}, {ps['both_cross_range'][1]:,}].")
        else:
            md_lines.append(f"- **Conditional Paired Crossing Difference**: Not estimable (0 pairs in `both-cross`).")
        if ps['both_cross_count'] > 0:
            md_lines.append(f"- **Directional Speed Advantage**: {ps['faster_ema_count']} of 5 seeds (in {ps['both_cross_faster_ema_count']} of {ps['both_cross_count']} uncensored pairs EMA crossed earlier, and in {ps['quadrant_counts']['EMA-only-cross']} pair(s) EMA crossed while baseline was censored).")
        else:
            md_lines.append(f"- **Directional Speed Advantage**: {ps['faster_ema_count']} of 5 seeds (in 0 uncensored pairs; in {ps['quadrant_counts']['EMA-only-cross']} pair(s) EMA crossed while baseline was censored).")
        md_lines.append(f"- **Final Validation Accuracy Difference (EMA - none)**: median = {ps['final_acc_diff_median_pp']:+.2f} pp, range = [{ps['final_acc_diff_range_pp'][0]:+.2f} pp, {ps['final_acc_diff_range_pp'][1]:+.2f} pp].")
        md_lines.append(f"- **Sign Consistency for Final Accuracy (EMA > none)**: {ps['higher_ema_count']} of 5 seeds.")
        md_lines.append("")

    with open(summary_md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Summary markdown report written to {summary_md_file}")

    # 4. Generate plots if matplotlib is available
    if HAS_MPL:
        generate_plots(data, arm_metrics, paired_rows)

    return per_arm_rows, paired_rows, paired_summary


def get_crossing_bar_props(m_arm, arm_color, default_cap=50000):
    """Returns visual bar properties distinguishing observed crossings from right-censored runs."""
    if m_arm.get("crossed"):
        return {
            "height": m_arm["cross_step"],
            "color": arm_color,
            "alpha": 0.85,
            "edgecolor": "none",
            "hatch": None,
            "linewidth": 0.0,
            "is_censored": False,
        }
    else:
        return {
            "height": default_cap,
            "color": "none",
            "alpha": 1.0,
            "edgecolor": arm_color,
            "hatch": "///",
            "linewidth": 1.5,
            "is_censored": True,
        }


def generate_plots(data, arm_metrics, paired_rows):
    """Generates visual comparisons for validation trajectories and crossing metrics."""
    print("Generating figures via matplotlib...")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # Plot 1: Validation trajectories grid (5 rows x 2 cols for wd=0 and wd=0.005)
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), sharex=True, sharey=True)
    fig.suptitle("Validation Accuracy Trajectories across 5 Seeds (100-update grid)\nSolid lines: EMA; Dashed lines: Baseline ('none')", fontsize=13)

    seeds_sorted = sorted(data.keys())
    for row_idx, s in enumerate(seeds_sorted):
        for col_idx, (wd_val, wd_str) in enumerate([(0.0, "0"), (0.005, "0.005")]):
            ax = axes[row_idx, col_idx]
            r_none = data[s][f"none_wd{wd_str}"]
            r_ema = data[s][f"ema_wd{wd_str}"]

            steps_none = [v["step"] for v in r_none["val_records"]]
            accs_none = [v["acc"] for v in r_none["val_records"]]
            steps_ema = [v["step"] for v in r_ema["val_records"]]
            accs_ema = [v["acc"] for v in r_ema["val_records"]]

            ax.plot(steps_none, accs_none, color="#d95f02", linestyle="--", linewidth=1.5, label="none (baseline)")
            ax.plot(steps_ema, accs_ema, color="#1b9e77", linestyle="-", linewidth=1.5, label="EMA")
            ax.axhline(0.99, color="#7570b3", linestyle=":", alpha=0.7, label="0.99 threshold")

            # Title & formatting
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"Seed {s} (wd={wd_val})", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Val Acc")
            if row_idx == 4:
                ax.set_xlabel("Updates")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right", fontsize=8)

    grid_fig_path = os.path.join(OUTPUT_DIR, "validation_curves_grid.png")
    fig.savefig(grid_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {grid_fig_path}")

    # Plot 2: Paired Crossing and Accuracy summary
    import matplotlib.patches as mpatches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Five-Seed Comparison: Grokfast-EMA vs Baseline", fontsize=13)

    # Subplot 1: Crossing Steps (wd=0.005 only, since wd=0 largely censors)
    seeds_labels = [f"Seed {s}" for s in seeds_sorted]
    x = range(len(seeds_sorted))
    width = 0.35

    has_censored = False
    for i, s in enumerate(seeds_sorted):
        m_none = arm_metrics[(s, "none_wd0.005")]
        m_ema = arm_metrics[(s, "ema_wd0.005")]

        pos_none = i - width / 2
        props_none = get_crossing_bar_props(m_none, "#d95f02", default_cap=50000)
        ax1.bar(
            pos_none,
            props_none["height"],
            width,
            color=props_none["color"],
            alpha=props_none["alpha"],
            edgecolor=props_none["edgecolor"],
            hatch=props_none["hatch"],
            linewidth=props_none["linewidth"],
        )
        if props_none["is_censored"]:
            has_censored = True
            ax1.scatter([pos_none], [50000], marker="x", color="#d95f02", s=50, zorder=5)

        pos_ema = i + width / 2
        props_ema = get_crossing_bar_props(m_ema, "#1b9e77", default_cap=50000)
        ax1.bar(
            pos_ema,
            props_ema["height"],
            width,
            color=props_ema["color"],
            alpha=props_ema["alpha"],
            edgecolor=props_ema["edgecolor"],
            hatch=props_ema["hatch"],
            linewidth=props_ema["linewidth"],
        )
        if props_ema["is_censored"]:
            has_censored = True
            ax1.scatter([pos_ema], [50000], marker="x", color="#1b9e77", s=50, zorder=5)

    ax1.axhline(50000, color="gray", linestyle=":", label="50k Update Cap")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(seeds_labels)
    ax1.set_ylabel("Updates to Sustained >=0.99 Crossing")
    ax1.set_title("Crossing Steps (wd = 0.005)")

    handles1 = [
        mpatches.Patch(facecolor="#d95f02", alpha=0.85, label="none_wd0.005"),
        mpatches.Patch(facecolor="#1b9e77", alpha=0.85, label="ema_wd0.005"),
        plt.Line2D([0], [0], color="gray", linestyle=":", label="50k Update Cap"),
    ]
    if has_censored:
        handles1.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=1.2,
                label="no crossing observed within 50k steps (censored)",
            )
        )
    ax1.legend(handles=handles1, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=0.6, fontsize=7.5)

    # Subplot 2: Final Accuracy Differences (EMA - none)
    diffs_wd0_pp = [(arm_metrics[(s, "ema_wd0")]["final_acc"] - arm_metrics[(s, "none_wd0")]["final_acc"]) * 100.0 for s in seeds_sorted]
    diffs_wd005_pp = [(arm_metrics[(s, "ema_wd0.005")]["final_acc"] - arm_metrics[(s, "none_wd0.005")]["final_acc"]) * 100.0 for s in seeds_sorted]

    ax2.plot(seeds_labels, diffs_wd0_pp, marker="o", linewidth=1.5, color="#7570b3", label="wd = 0.0")
    ax2.plot(seeds_labels, diffs_wd005_pp, marker="s", linewidth=1.5, color="#e7298a", label="wd = 0.005")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Final Val Acc Diff: EMA - none (pp)")
    ax2.set_title("Final Accuracy Difference (Step 50,000)")
    ax2.legend(fontsize=8)

    summary_fig_path = os.path.join(OUTPUT_DIR, "crossing_and_accuracy_comparison.png")
    fig.savefig(summary_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {summary_fig_path}")


def main():
    print("Starting Five-Seed Analysis Pipeline...")
    data = validate_and_load_records()
    run_part_b_prefix_check()
    per_arm_rows, paired_rows, paired_summary = run_part_c_analysis(data)
    print("\nAnalysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
