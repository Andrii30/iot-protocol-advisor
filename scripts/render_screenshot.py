"""Render docs/screenshot.png from real advisor output.

The GUI can't run in CI, so this draws the same window (toolbar, summary line,
results table, status bar) with Pillow using the actual verdicts produced for
examples/large_dataset.csv. Run:  python scripts/render_screenshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from protocol_advisor import Engine  # noqa: E402
from protocol_advisor.advisor import advise  # noqa: E402
from protocol_advisor.sources import CsvSource  # noqa: E402

SCALE = 2
W, H = 1180, 620
VERDICT_FG = {"KEEP": "#1b7f2e", "SWITCH": "#e2691a", "KEEP_LOW_CONFIDENCE": "#6b6b6b"}
COLS = [("Device", 140), ("Current", 82), ("Recommended", 108), ("Confidence", 88),
        ("Verdict", 158), ("Rule-based", 98), ("ML=rule", 64), ("Samples", 62),
        ("Top factor", 150), ("2nd factor", 150)]


def _font(name, size):
    for p in (f"/System/Library/Fonts/{name}", f"/Library/Fonts/{name}",
              f"/System/Library/Fonts/Supplemental/{name}"):
        if Path(p).exists():
            return ImageFont.truetype(p, size * SCALE)
    return ImageFont.load_default()


def main() -> None:
    eng = Engine()
    info = eng.load_or_train()
    import pandas as pd
    full = advise(CsvSource(str(REPO / "examples" / "large_dataset.csv")).load(), eng)
    total_devices = len(full)
    kc = int((full["verdict"] == "KEEP").sum())
    sc = int((full["verdict"] == "SWITCH").sum())
    lc = int((full["verdict"] == "KEEP_LOW_CONFIDENCE").sum())
    # Show a realistic mix of verdicts in the table, not one block.
    report = pd.concat([
        full[full["verdict"] == "SWITCH"].head(9),
        full[full["verdict"] != "SWITCH"].head(9),
    ]).sort_values("device_id").reset_index(drop=True)

    img = Image.new("RGB", (W * SCALE, H * SCALE), "#ececec")
    d = ImageDraw.Draw(img)
    s = SCALE
    reg = _font("Helvetica.ttc", 13)
    bold = _font("Helvetica.ttc", 13)
    big = _font("Helvetica.ttc", 15)
    small = _font("Helvetica.ttc", 11)

    # title bar
    d.rectangle([0, 0, W * s, 30 * s], fill="#e4e4e4")
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        d.ellipse([(16 + i * 20) * s, 10 * s, (26 + i * 20) * s, 20 * s], fill=c)
    d.text((W / 2 * s, 15 * s), "IoT Protocol Advisor", font=bold, fill="#333", anchor="mm")

    # toolbar
    d.rectangle([0, 30 * s, W * s, 72 * s], fill="#f4f4f4")
    x = 14
    for label in ("Open data file…", "Connect to PostgreSQL…", "Retrain model", "Export report…"):
        w = d.textlength(label, font=reg) / s + 24
        d.rounded_rectangle([x * s, 41 * s, (x + w) * s, 63 * s], radius=5 * s,
                            fill="#fdfdfd", outline="#c4c4c4", width=s)
        d.text(((x + 12) * s, 52 * s), label, font=reg, fill="#222", anchor="lm")
        x += w + 8

    # summary line
    d.text((16 * s, 90 * s),
           f"{total_devices} devices from large_dataset.csv   ·   KEEP: {kc}, SWITCH: {sc}, "
           f"KEEP_LOW_CONFIDENCE: {lc}", font=big, fill="#111", anchor="lm")

    # table
    top = 108
    d.rectangle([8 * s, top * s, (W - 8) * s, (H - 34) * s], fill="#ffffff", outline="#cfcfcf", width=s)
    cx = 12
    d.rectangle([8 * s, top * s, (W - 8) * s, (top + 26) * s], fill="#f0f0f0")
    for name, cw in COLS:
        d.text((cx * s, (top + 13) * s), name, font=bold, fill="#333", anchor="lm")
        cx += cw
    ry = top + 26
    for _, row in report.head(17).iterrows():
        cx = 12
        vals = [str(row["device_id"]), str(row["current_protocol"]),
                str(row["recommended_protocol"]), f"{row['confidence']:.2f}",
                str(row["verdict"]), str(row["rule_based"]),
                "yes" if row["ml_agrees_rule"] else "no",
                str(row["n_samples"]),
                str(row["factor_1"]), str(row["factor_2"])]
        for (name, cw), v in zip(COLS, vals):
            fg = VERDICT_FG.get(row["verdict"], "#222") if name == "Verdict" else "#222"
            d.text((cx * s, (ry + 13) * s), v, font=reg, fill=fg, anchor="lm")
            cx += cw
        d.line([8 * s, (ry + 26) * s, (W - 8) * s, (ry + 26) * s], fill="#eee", width=s)
        ry += 26

    # status bar
    d.rectangle([0, (H - 26) * s, W * s, H * s], fill="#e9e9e9")
    d.text((10 * s, (H - 13) * s),
           f"Model: {info.model_name} · holdout macro-F1: {info.macro_f1:.2f} · "
           f"accuracy: {info.accuracy:.2f} · sklearn {info.sklearn_version}",
           font=small, fill="#444", anchor="lm")

    out = REPO / "docs" / "screenshot.png"
    img.resize((W, H), Image.LANCZOS).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
