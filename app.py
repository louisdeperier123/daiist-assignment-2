"""
Assignment 2: Gradio dashboard for the spray / review / don't-spray decision.

    uv run python main.py app

Never trains anything. It loads the committed weights in weights/ and the
outputs train.py saved in artifacts/, and recomputes costs live with the same
formulas train.py used (costing.py).
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

from costing import (Costs, confusion_per_1000, expected_cost_per_plant,
                     outcome_rates, reference_policies, review_share)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = sorted((ROOT / "weights").glob("*.pt"))
COSTS = Costs()


def load_json(name, default=None):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else default


def load_csv(name, columns):
    p = ART / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=columns)


# ---------------------------------------------------------- artifacts ---
METRICS = load_json("metrics.json", {})
OP = load_json("operating_point.json", {})
STATS = load_json("dataset_stats.json", {})
SOURCE = load_json("dataset_source.json", {})
SWEEP = load_csv("cost_sweep.csv", ["threshold", "cost_per_1000", "weed_recall", "crop_sprayed_rate"])
PLANTS = load_csv("plant_scores.csv", ["image", "tray", "label", "score"])
SOIL = load_csv("soil_scores.csv", ["image", "score"])
WEED_S = PLANTS.loc[PLANTS.label == "weed", "score"].to_numpy()
CROP_S = PLANTS.loc[PLANTS.label == "crop", "score"].to_numpy()
SOIL_S = SOIL["score"].to_numpy()
N_PLANTS = len(PLANTS)

T_STAR = float(OP.get("model_alone_threshold", 0.5))
BAND = OP.get("review_band_capacity_limited", {})
T_LOW0 = float(BAND.get("t_low", max(T_STAR - 0.1, 0.01)))
T_HIGH0 = float(BAND.get("t_high", T_STAR))
if T_LOW0 >= T_HIGH0:  # give the sliders a visible band to start from
    T_LOW0 = max(round(T_HIGH0 - 0.10, 2), 0.01)

MODEL = None
if WEIGHTS:
    from ultralytics import YOLO
    MODEL = YOLO(str(WEIGHTS[0]))
NAMES = dict(MODEL.names) if MODEL else {}

COL_SPRAY, COL_REVIEW, COL_CROP = (220, 40, 40), (245, 160, 0), (30, 160, 70)


# ------------------------------------------------------------ helpers ---
def predict_image(img, t_low, t_high):
    if img is None:
        return None, "Upload an image or pick an example."
    if MODEL is None:
        return img, "No weights found in weights/."
    t_low = min(t_low, t_high)
    res = MODEL.predict(img, conf=0.01, verbose=False)[0]
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    counts = {"spray": 0, "review": 0, "crop kept": 0}
    for cls, conf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                              res.boxes.xyxy.tolist()):
        name = NAMES.get(int(cls), str(cls))
        if name == "weed" and conf >= t_high:
            col, tag = COL_SPRAY, "spray"
        elif name == "weed" and conf >= t_low:
            col, tag = COL_REVIEW, "review"
        elif name == "crop" and conf >= 0.25:
            col, tag = COL_CROP, "crop kept"
        else:
            continue
        counts[tag] += 1
        draw.rectangle(box, outline=col, width=3)
        draw.text((box[0] + 2, max(box[1] - 11, 0)), f"{name} {conf:.2f}", fill=col)
    summary = (f"**Spray:** {counts['spray']} (red)  ·  **Review:** {counts['review']} (orange)  ·  "
               f"**Crop detected, not sprayed:** {counts['crop kept']} (green)")
    return out, summary


def band_outputs(t_low, t_high):
    t_low = min(t_low, t_high)
    if N_PLANTS == 0:
        empty = pd.DataFrame()
        return empty, go.Figure(), "Run `main.py train` first: artifacts/ is empty.", cost_curve(t_low, t_high)
    r = outcome_rates(WEED_S, CROP_S, SOIL_S, N_PLANTS, t_low, t_high)
    cm = confusion_per_1000(r, COSTS)
    cm_df = pd.DataFrame(np.round(cm, 1), columns=["Spray", "Human review", "Don't spray"],
                         index=["Actual weed", "Actual crop", "Bare soil (false detection)"]
                         ).reset_index(names="per 1,000 plants")

    fig = go.Figure(go.Heatmap(z=cm, x=["Spray", "Human review", "Don't spray"],
                               y=["Actual weed", "Actual crop", "Bare soil"],
                               text=np.round(cm, 1), texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Decisions per 1,000 plants (production base rate)",
                      yaxis_autorange="reversed", height=320, margin=dict(t=50, b=30))

    cost = 1000 * expected_cost_per_plant(r, COSTS)
    share = review_share(r, COSTS)
    reviews_day = share * COSTS.plants_per_day
    hours = reviews_day * COSTS.review_seconds / 3600
    people = hours / COSTS.review_hours_per_day
    alone = 1000 * expected_cost_per_plant(outcome_rates(WEED_S, CROP_S, SOIL_S, N_PLANTS,
                                                         T_STAR, T_STAR), COSTS)
    policies = {k: 1000 * v for k, v in reference_policies(COSTS).items()}
    policies[f"Model alone (t = {T_STAR:.2f})"] = alone
    policies["This band"] = cost
    fit = "within" if people <= COSTS.reviewers_available else "**over**"
    text = (
        f"### Expected cost: €{cost:.2f} per 1,000 plants\n"
        f"- Sent to human review: **{100 * share:.2f}%** of plants, "
        f"about **{reviews_day:,.0f} checks/day** over {COSTS.hectares_per_day:.0f} ha\n"
        f"- At {COSTS.review_seconds:.1f} s each: **{hours:,.1f} h/day**, "
        f"**{people:.1f} reviewers** ({fit} the {COSTS.reviewers_available} available)\n"
        f"- Weed recall (sprayed or reviewed): {100 * (r['weed'][0] + r['weed'][1]):.1f}%  ·  "
        f"Crops sprayed: {100 * r['crop'][0]:.2f}%\n\n"
        "| Policy | € per 1,000 plants |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.2f} |" for k, v in policies.items())
    )
    return cm_df, fig, text, cost_curve(t_low, t_high)


def cost_curve(t_low=None, t_high=None):
    fig = go.Figure()
    if len(SWEEP):
        fig.add_trace(go.Scatter(x=SWEEP.threshold, y=SWEEP.cost_per_1000, mode="lines",
                                 name="Model alone"))
        y_star = float(SWEEP.loc[(SWEEP.threshold - T_STAR).abs().idxmin(), "cost_per_1000"])
        fig.add_trace(go.Scatter(x=[T_STAR], y=[y_star], mode="markers+text",
                                 marker=dict(size=12, color="black", symbol="star"),
                                 text=[f"chosen {T_STAR:.2f}"], textposition="top center",
                                 name="Chosen threshold"))
        for name, v in reference_policies(COSTS).items():
            fig.add_hline(y=1000 * v, line_dash="dot", annotation_text=name,
                          annotation_position="top left", opacity=0.5)
    if t_low is not None:
        fig.add_vrect(x0=min(t_low, t_high), x1=t_high, fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="review band")
    fig.update_layout(title="Expected cost vs confidence threshold",
                      xaxis_title="Weed confidence threshold",
                      yaxis_title="€ per 1,000 plants", height=380, margin=dict(t=50))
    return fig


def per_class_table():
    off = METRICS.get("official_test", {})
    rows = [{"Class": k, **{m: round(v, 3) for m, v in d.items()}}
            for k, d in off.get("per_class", {}).items()]
    if off:
        rows.append({"Class": "all (headline)", "AP50": round(off["mAP50"], 3),
                     "mAP50_95": round(off["mAP50_95"], 3)})
    return pd.DataFrame(rows)


def baseline_table():
    rows = []
    for name, d in METRICS.get("baselines", {}).items():
        rows.append({"Model": name, "AP50 weed": round(d["AP50_weed"], 3),
                     "AP50 crop": round(d["AP50_crop"], 3),
                     "Best threshold": d["best_threshold"],
                     "Weed recall": round(d["weed_recall"], 3),
                     "Crops sprayed": round(d["crop_sprayed_rate"], 3),
                     "€ / 1,000 plants": round(d["cost_per_1000"], 2)})
    return pd.DataFrame(rows)


def confusion_fig():
    off = METRICS.get("official_test", {})
    if not off:
        return go.Figure()
    cm = np.array(off["confusion_matrix"]); labels = off["confusion_labels"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=cm.astype(int),
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Detection confusion matrix (test, conf 0.25)",
                      xaxis_title="True", yaxis_title="Predicted",
                      yaxis_autorange="reversed", height=360, margin=dict(t=50))
    return fig


def subset_table():
    return pd.DataFrame([{"Photos": k, **v} for k, v in OP.get("subsets", {}).items()])


def dataset_table():
    rows = []
    for split, s in STATS.items():
        for cls, c in s["classes"].items():
            rows.append({"Split": split, "Images": s["images"], "Class": cls,
                         "Instances": c["instances"], "Images containing": c["images_containing"],
                         "In tray photos": c["instances_in_tray_images"],
                         "Median side (px)": round(c["median_side_px_at_640"], 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- UI ---
examples = [str(p) for p in sorted((ART / "examples").glob("*"))]
src = (f"{SOURCE.get('workspace')}/{SOURCE.get('project')} v{SOURCE.get('version')}"
       if SOURCE else "unknown")

with gr.Blocks(title="Crop vs weed: spray decision") as demo:
    gr.Markdown(
        "# Crop vs weed: spray, review, or leave\n"
        f"Positive class: **weed** (spray). Dataset: `{src}`. "
        f"Weights: `{WEIGHTS[0].name if WEIGHTS else 'none'}`. "
        f"Base rate assumed: {COSTS.base_rate_weed:.0%} weeds. "
        f"Costs: crop sprayed €{COSTS.fp_crop_sprayed}, weed missed €{COSTS.fn_weed_missed}, "
        f"dose €{COSTS.tp_weed_sprayed}, review €{COSTS.review_cost:.4f}.")

    with gr.Row():
        t_low = gr.Slider(0.01, 0.99, value=T_LOW0, step=0.01,
                          label="Lower threshold: below this, don't spray")
        t_high = gr.Slider(0.01, 0.99, value=T_HIGH0, step=0.01,
                           label="Upper threshold: at or above this, spray")

    with gr.Tab("Decision & cost"):
        with gr.Row():
            with gr.Column():
                cm_plot = gr.Plot()
                cm_table = gr.Dataframe(interactive=False)
            with gr.Column():
                summary = gr.Markdown()
        curve = gr.Plot()

    with gr.Tab("Try an image"):
        with gr.Row():
            inp = gr.Image(type="pil", label="Field image")
            out = gr.Image(type="pil", label="Decisions at current thresholds")
        pred_text = gr.Markdown()
        if examples:
            gr.Examples(examples, inputs=inp)

    with gr.Tab("Per-class results"):
        gr.Markdown("### Finetuned model on the test split (Ultralytics, per class)")
        gr.Dataframe(per_class_table(), interactive=False)
        gr.Plot(confusion_fig())
        gr.Markdown("### Against the baselines (same test split, each at its own best threshold)")
        gr.Dataframe(baseline_table(), interactive=False)
        gr.Markdown("### Tray photos vs open-field photos, at the chosen threshold")
        gr.Dataframe(subset_table(), interactive=False)

    with gr.Tab("Dataset"):
        gr.Dataframe(dataset_table(), interactive=False)

    for s in (t_low, t_high):
        s.change(band_outputs, [t_low, t_high], [cm_table, cm_plot, summary, curve])
        s.change(predict_image, [inp, t_low, t_high], [out, pred_text])
    inp.change(predict_image, [inp, t_low, t_high], [out, pred_text])
    demo.load(band_outputs, [t_low, t_high], [cm_table, cm_plot, summary, curve])


if __name__ == "__main__":
    demo.launch()
