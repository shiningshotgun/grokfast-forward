#!/usr/bin/env python3
"""Plot validation accuracy and validation loss for the four Grokfast study arms.

Reads only the saved full-set validation records in
study_outputs/approved_comparison/arm_*.json and writes two figures to figures/.
No training, no recomputation of study metrics, no network access.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "study_outputs" / "approved_comparison"
OUT_DIR = ROOT / "figures"

ARMS = [
    ("none_wd0", "baseline, wd 0", "#7a7a7a", True),
    ("ema_wd0", "EMA, wd 0", "#d97706", True),
    ("none_wd0.005", "baseline, wd 0.005", "#2563eb", False),
    ("ema_wd0.005", "EMA, wd 0.005", "#dc2626", False),
]

EXPECTED = {
    "none_wd0": {
        "rows": 501,
        "last_step": 50000,
        "last_acc": 0.9390034364261168,
        "first": None,
    },
    "ema_wd0": {
        "rows": 420,
        "last_step": 41900,
        "last_acc": 0.613831615120275,
        "first": None,
    },
    "none_wd0.005": {"rows": 464, "last_step": 46300, "last_acc": 1.0, "first": 5900},
    "ema_wd0.005": {"rows": 439, "last_step": 43800, "last_acc": 1.0, "first": 1200},
}


def load_arm(key):
    path = DATA_DIR / ("arm_" + key + ".json")
    with path.open() as f:
        data = json.load(f)
    all_rows = data["val_records"]
    full = [
        r
        for r in all_rows
        if r.get("complete") is True
        and r.get("full_set") is True
        and r.get("n") == 4656
    ]
    partial = [
        r
        for r in all_rows
        if not (
            r.get("complete") is True
            and r.get("full_set") is True
            and r.get("n") == 4656
        )
    ]
    return data, full, partial


def first_crossing(rows, threshold=0.95):
    for r in rows:
        if r["acc"] >= threshold:
            return r["step"]
    return None


def verify():
    loaded = {}
    for key, label, color, dashed in ARMS:
        data, full, partial = load_arm(key)
        exp = EXPECTED[key]
        assert len(full) == exp["rows"], (
            key + ": full rows " + str(len(full)) + " != " + str(exp["rows"])
        )
        assert full[-1]["step"] == exp["last_step"], (
            key
            + ": last step "
            + str(full[-1]["step"])
            + " != "
            + str(exp["last_step"])
        )
        assert full[-1]["acc"] == exp["last_acc"], (
            key + ": last acc " + str(full[-1]["acc"]) + " != " + str(exp["last_acc"])
        )
        fc = first_crossing(full)
        assert fc == exp["first"], (
            key + ": first crossing " + str(fc) + " != " + str(exp["first"])
        )
        bad = [r for r in full if not (r["loss"] > 0)]
        assert not bad, key + ": non-positive loss at " + str([r["step"] for r in bad])
        loaded[key] = (data, full, partial)
        print(
            key
            + ": "
            + str(len(full))
            + " full rows, last step "
            + str(full[-1]["step"])
            + ", acc "
            + format(full[-1]["acc"] * 100, ".4f")
            + "%, first>=95% "
            + ("none" if fc is None else str(fc))
        )
    _, _, partial = loaded["none_wd0.005"]
    assert len(partial) == 1, "none_wd0.005: expected 1 partial row, got " + str(
        len(partial)
    )
    assert partial[0]["step"] == 46400 and partial[0]["n"] == 1536, (
        "none_wd0.005: partial row "
        + str(partial[0]["step"])
        + "/"
        + str(partial[0]["n"])
    )
    for key in ("none_wd0", "ema_wd0", "ema_wd0.005"):
        _, _, partial = loaded[key]
        assert len(partial) == 0, (
            key + ": expected 0 partial rows, got " + str(len(partial))
        )
    return loaded


def xscale(step, xmax, left, width):
    return left + (step / xmax) * width


def yscale_acc(v, ymin, ymax, top, height):
    return top + (1.0 - (v - ymin) / (ymax - ymin)) * height


def yscale_log(v, ymin, ymax, top, height):
    import math

    return (
        top
        + (
            1.0
            - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
        )
        * height
    )


def polyline_points(
    rows, xkey, yval_func, xmax, left, width, top, height, ymin, ymax, log=False
):
    pts = []
    for r in rows:
        x = xscale(r[xkey], xmax, left, width)
        if log:
            y = yscale_log(yval_func(r), ymin, ymax, top, height)
        else:
            y = yscale_acc(yval_func(r), ymin, ymax, top, height)
        pts.append((x, y))
    return " ".join("{:.2f},{}".format(p[0], "{:.2f}".format(p[1])) for p in pts)


def build_svg(title, y_mode, loaded, xmax):
    W, H = 1000, 560
    left, right = 80, 800
    top, bottom = 55, 490
    width = right - left
    height = bottom - top

    if y_mode == "acc":
        ymin, ymax = -2.0, 102.0
        yticks = [0, 20, 40, 60, 80, 100]
        yfmt = lambda v: str(int(v))
    else:
        ymin, ymax = 0.003, 40.0
        yticks = [0.01, 0.1, 1, 10]
        yfmt = lambda v: "{:.2g}".format(v) if v < 1 else str(int(v))

    xticks = [0, 10000, 20000, 30000, 40000, 50000]

    parts = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}">'.format(W, H, W, H)
    )
    parts.append(
        '<rect x="0" y="0" width="{}" height="{}" fill="#ffffff"/>'.format(W, H)
    )

    for xt in xticks:
        x = xscale(xt, xmax, left, width)
        parts.append(
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" '
            'stroke="#e5e7eb" stroke-width="0.8"/>'.format(x, top, x, bottom)
        )
    for yt in yticks:
        if y_mode == "acc":
            y = yscale_acc(yt, ymin, ymax, top, height)
        else:
            y = yscale_log(yt, ymin, ymax, top, height)
        parts.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" '
            'stroke="#e5e7eb" stroke-width="0.8"/>'.format(left, y, right, y)
        )

    parts.append(
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#374151" stroke-width="1.2"/>'.format(
            left, bottom, right, bottom
        )
    )
    parts.append(
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#374151" stroke-width="1.2"/>'.format(
            left, top, left, bottom
        )
    )

    for xt in xticks:
        x = xscale(xt, xmax, left, width)
        parts.append(
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#374151" stroke-width="1"/>'.format(
                x, bottom, x, bottom + 4
            )
        )
        label = str(xt) if xt < 1000 else str(xt // 1000) + "k"
        if xt == 0:
            label = "0"
        parts.append(
            '<text x="{:.2f}" y="{}" font-size="11" text-anchor="middle" '
            'font-family="sans-serif" fill="#374151">{}</text>'.format(
                x, bottom + 18, label
            )
        )

    for yt in yticks:
        if y_mode == "acc":
            y = yscale_acc(yt, ymin, ymax, top, height)
        else:
            y = yscale_log(yt, ymin, ymax, top, height)
        parts.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#374151" stroke-width="1"/>'.format(
                left - 4, y, left, y
            )
        )
        parts.append(
            '<text x="{}" y="{:.2f}" font-size="11" text-anchor="end" '
            'dominant-baseline="middle" font-family="sans-serif" fill="#374151">{}</text>'.format(
                left - 8, y, yfmt(yt)
            )
        )

    if y_mode == "acc":
        y95 = yscale_acc(95.0, ymin, ymax, top, height)
        parts.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" '
            'stroke="#6b7280" stroke-width="0.9"/>'.format(left, y95, right, y95)
        )
        parts.append(
            '<text x="{}" y="{:.2f}" font-size="10" text-anchor="end" '
            'dominant-baseline="middle" font-family="sans-serif" fill="#6b7280">95%</text>'.format(
                right - 2, y95
            )
        )

    for key, label, color, dashed in ARMS:
        data, full, partial = loaded[key]
        if y_mode == "acc":
            pts = polyline_points(
                full,
                "step",
                lambda r: r["acc"] * 100.0,
                xmax,
                left,
                width,
                top,
                height,
                ymin,
                ymax,
                log=False,
            )
        else:
            pts = polyline_points(
                full,
                "step",
                lambda r: r["loss"],
                xmax,
                left,
                width,
                top,
                height,
                ymin,
                ymax,
                log=True,
            )
        dash_attr = ' stroke-dasharray="7,4"' if dashed else ""
        parts.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="1.2"{} '
            'stroke-linejoin="round" stroke-linecap="round"/>'.format(
                pts, color, dash_attr
            )
        )
        last = full[-1]
        if y_mode == "acc":
            lx = xscale(last["step"], xmax, left, width)
            ly = yscale_acc(last["acc"] * 100.0, ymin, ymax, top, height)
        else:
            lx = xscale(last["step"], xmax, left, width)
            ly = yscale_log(last["loss"], ymin, ymax, top, height)
        parts.append(
            '<circle cx="{:.2f}" cy="{:.2f}" r="3" fill="{}"/>'.format(lx, ly, color)
        )
        if y_mode == "acc":
            fc = first_crossing(full)
            if fc is not None:
                crow = next(r for r in full if r["step"] == fc)
                cx = xscale(fc, xmax, left, width)
                cy = yscale_acc(crow["acc"] * 100.0, ymin, ymax, top, height)
                parts.append(
                    '<circle cx="{:.2f}" cy="{:.2f}" r="5" fill="none" '
                    'stroke="{}" stroke-width="1.5"/>'.format(cx, cy, color)
                )

    lx0 = 820
    ly0 = 70
    parts.append(
        '<text x="{}" y="{}" font-size="12" font-weight="bold" '
        'font-family="sans-serif" fill="#374151">Legend</text>'.format(lx0, ly0 - 12)
    )
    for i, (key, label, color, dashed) in enumerate(ARMS):
        yy = ly0 + i * 22
        dash_attr = ' stroke-dasharray="7,4"' if dashed else ""
        parts.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1.2"{} '
            'stroke-linecap="round"/>'.format(lx0, yy, lx0 + 26, yy, color, dash_attr)
        )
        parts.append(
            '<circle cx="{}" cy="{}" r="2.5" fill="{}"/>'.format(lx0 + 26, yy, color)
        )
        parts.append(
            '<text x="{}" y="{}" font-size="11" dominant-baseline="middle" '
            'font-family="sans-serif" fill="#374151">{}</text>'.format(
                lx0 + 34, yy, label
            )
        )

    parts.append(
        '<text x="{}" y="30" font-size="15" text-anchor="middle" '
        'font-family="sans-serif" fill="#111827" font-weight="bold">{}</text>'.format(
            W // 2, title
        )
    )

    parts.append(
        '<text x="{}" y="{}" font-size="12" text-anchor="middle" '
        'font-family="sans-serif" fill="#374151">Update</text>'.format(
            (left + right) // 2, bottom + 46
        )
    )
    if y_mode == "acc":
        ylab = "Validation accuracy (%)"
    else:
        ylab = "Validation loss (cross-entropy, log scale)"
    parts.append(
        '<text x="22" y="{}" font-size="12" text-anchor="middle" '
        'font-family="sans-serif" fill="#374151" '
        'transform="rotate(-90, 22, {})">{}</text>'.format(
            (top + bottom) // 2, (top + bottom) // 2, ylab
        )
    )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_svg(loaded):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xmax = 50000
    svg1 = build_svg(
        "Validation accuracy, full validation set (n = 4656)", "acc", loaded, xmax
    )
    (OUT_DIR / "validation_accuracy.svg").write_text(svg1, encoding="utf-8")
    svg2 = build_svg(
        "Validation loss, full validation set (n = 4656)", "loss", loaded, xmax
    )
    (OUT_DIR / "validation_loss.svg").write_text(svg2, encoding="utf-8")


def generate_png(loaded):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xmax = 50000

    def draw(ax, y_mode):
        for key, label, color, dashed in ARMS:
            data, full, partial = loaded[key]
            xs = [r["step"] for r in full]
            if y_mode == "acc":
                ys = [r["acc"] * 100.0 for r in full]
            else:
                ys = [r["loss"] for r in full]
            ls = "--" if dashed else "-"
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.2, label=label)
            ax.plot(
                xs[-1], ys[-1], marker="o", color=color, markersize=4, linestyle="None"
            )
            if y_mode == "acc":
                fc = first_crossing(full)
                if fc is not None:
                    crow = next(r for r in full if r["step"] == fc)
                    ax.plot(
                        fc,
                        crow["acc"] * 100.0,
                        marker="o",
                        color=color,
                        markerfacecolor="none",
                        markersize=7,
                        linestyle="None",
                    )
        if y_mode == "acc":
            ax.set_ylim(-2, 102)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.axhline(95, color="#6b7280", linewidth=0.9)
            ax.text(
                xmax, 95, "95%", color="#6b7280", fontsize=9, va="center", ha="right"
            )
            ax.set_ylabel("Validation accuracy (%)")
            ax.set_title("Validation accuracy, full validation set (n = 4656)")
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.003, 40)
            ax.set_yticks([0.01, 0.1, 1, 10])
            ax.set_ylabel("Validation loss (cross-entropy, log scale)")
            ax.set_title("Validation loss, full validation set (n = 4656)")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Update")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color="#e5e7eb")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw(ax, "acc")
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "validation_accuracy.png", dpi=150, metadata={"Software": None}
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw(ax, "loss")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_loss.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def main():
    loaded = verify()
    try:
        import matplotlib  # noqa: F401

        HAS_MPL = True
    except Exception:
        HAS_MPL = False
    if HAS_MPL:
        generate_png(loaded)
    else:
        generate_svg(loaded)


if __name__ == "__main__":
    main()
