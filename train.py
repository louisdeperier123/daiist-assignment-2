"""
Assignment 2: training pipeline (crop vs weed, targeted herbicide spraying).

    uv run python main.py train

Start to finish, with no manual steps:
  1. Download the Roboflow Universe dataset (key via api_key.load_api_key).
  2. Describe it: instances per class, split sizes, object size.
  3. Finetune yolo11n from the COCO-pretrained checkpoint; copy best.pt to weights/.
  4. Evaluate on the test split: finetuned model, COCO model zero-shot, and a
     trivial predict-nothing baseline. Per-class P/R/AP and confusion matrix.
  5. Turn detections into one spray decision per plant, sweep the confidence
     threshold, cost each point with costing.py, and pick the operating point
     (model alone) and the review band (model + human).
  6. Save everything app.py needs into artifacts/ so the app never retrains.

Optional environment overrides: EPOCHS, IMGSZ, BATCH, MODEL, DEVICE.
LOCAL_DATASET_DIR=<folder with data.yaml> skips the download (offline testing only).
"""

import json
import os
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from costing import (Costs, confusion_per_1000, expected_cost_per_plant,
                     outcome_rates, reference_policies, review_share)

# ----------------------------------------------------------------- config ---
WORKSPACE = "object-detection"
PROJECT = "weed-crop-aerial"
VERSION = None            # None = latest published version; set an int to pin it

MODEL = os.environ.get("MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", 100))
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
DEVICE = os.environ.get("DEVICE") or None
SEED = 0

MATCH_IOU = 0.3           # a detection "hits" a plant at IoU >= 0.3 (tiny boxes jitter)
POSITIVE = "weed"
NEGATIVE = "crop"
COSTS = Costs()
N_BOOTSTRAP = 1000

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
ART = ROOT / "artifacts"
RUNS = ROOT / "runs"


def log(msg):
    print(f"\n[train] {msg}", flush=True)


# --------------------------------------------------------------- 1. data ---
def download_dataset() -> Path:
    local = os.environ.get("LOCAL_DATASET_DIR")
    if local:
        log(f"Using local dataset at {local} (download skipped)")
        root = Path(local).resolve()
        version = "local"
    else:
        from roboflow import Roboflow
        from api_key import load_api_key
        rf = Roboflow(api_key=load_api_key())
        project = rf.workspace(WORKSPACE).project(PROJECT)
        version = VERSION
        if version is None:
            version = max(int(str(v.version).split("/")[-1]) for v in project.versions())
        log(f"Downloading {WORKSPACE}/{PROJECT} version {version}")
        ds = project.version(version).download(
            "yolov11", location=str(DATA_DIR / f"{PROJECT}-v{version}"), overwrite=False)
        root = Path(ds.location).resolve()

    # Roboflow's data.yaml uses relative paths that break depending on cwd.
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    fixed = {"path": str(root), "train": "train/images", "val": "valid/images",
             "test": "test/images", "nc": len(names), "names": names}
    out = root / "data_fixed.yaml"
    out.write_text(yaml.safe_dump(fixed))
    (ART / "dataset_source.json").write_text(json.dumps(
        {"workspace": WORKSPACE, "project": PROJECT, "version": version}, indent=2))
    return out


def read_split(data_yaml: Path, split: str):
    """List of (image_path, gt) where gt is an array of [cls, x1, y1, x2, y2] in pixels."""
    cfg = yaml.safe_load(data_yaml.read_text())
    img_dir = Path(cfg["path"]) / {"train": "train", "val": "valid", "test": "test"}[split] / "images"
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        w, h = Image.open(img).size
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        rows = []
        if lab.exists():
            for line in lab.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:  # boxes only; polygons are not expected here
                    c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                    rows.append([c, (x - bw / 2) * w, (y - bh / 2) * h,
                                 (x + bw / 2) * w, (y + bh / 2) * h])
        items.append((img, np.array(rows, dtype=float).reshape(-1, 5), (w, h)))
    return items


def is_tray(path: Path) -> bool:
    """Source tag: planter-tray photos are named IMG_*, open-field photos are numeric."""
    return path.name.startswith("IMG_")


# ----------------------------------------------------------- 2. describe ---
def describe(data_yaml: Path, names):
    log("Describing the dataset")
    stats = {}
    for split in ["train", "val", "test"]:
        items = read_split(data_yaml, split)
        s = {"images": len(items), "classes": {}}
        for ci, name in enumerate(names):
            areas, imgs, tray = [], 0, 0
            for path, gt, (w, h) in items:
                m = gt[:, 0] == ci
                if m.any():
                    imgs += 1
                    tray += int(m.sum()) if is_tray(path) else 0
                    areas += list((gt[m, 3] - gt[m, 1]) * (gt[m, 4] - gt[m, 2]) / (w * h))
            if not areas:
                continue
            a = np.array(areas)
            s["classes"][name] = {
                "instances": len(a), "images_containing": imgs,
                "instances_in_tray_images": tray,
                "rel_area_median": float(np.median(a)),
                "rel_area_p10": float(np.percentile(a, 10)),
                "rel_area_p90": float(np.percentile(a, 90)),
                "median_side_px_at_640": float(np.sqrt(np.median(a)) * 640),
            }
        s["tray_images"] = sum(is_tray(p) for p, _, _ in items)
        stats[split] = s
    (ART / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        print(k, v["images"], "images,", {n: c["instances"] for n, c in v["classes"].items()})
    return stats


# -------------------------------------------------------------- 3. train ---
def train(data_yaml: Path) -> Path:
    from ultralytics import YOLO
    log(f"Finetuning {MODEL}: epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH}")
    model = YOLO(MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                patience=25, seed=SEED, deterministic=True, device=DEVICE,
                project=str(RUNS), name="finetune", exist_ok=True, plots=True)
    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for old in WEIGHTS_DIR.glob("*.pt"):
        old.unlink()
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy(best, target)
    log(f"Saved {target}")
    return target


# ----------------------------------------------------------- 4. evaluate ---
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def predict(model, items, batch=16):
    """Low-confidence predictions (conf 0.001) so every threshold can be swept later."""
    out = []
    paths = [str(p) for p, _, _ in items]
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], conf=0.001, iou=0.6, imgsz=IMGSZ,
                               max_det=300, verbose=False, device=DEVICE):
            b = r.boxes
            out.append(np.column_stack([b.cls.cpu().numpy(), b.xyxy.cpu().numpy(),
                                        b.conf.cpu().numpy()]).reshape(-1, 6))
    return out


def ap50(items, preds, gt_cls, pred_cls_set):
    """VOC-style all-point AP at IoU 0.5 for one ground-truth class."""
    recs, n_gt = [], 0
    for (_, gt, _), pr in zip(items, preds):
        g = gt[gt[:, 0] == gt_cls, 1:]
        n_gt += len(g)
        p = pr[np.isin(pr[:, 0], list(pred_cls_set))]
        p = p[np.argsort(-p[:, 5])]
        used = np.zeros(len(g), bool)
        ious = iou_matrix(p[:, 1:5], g)
        for k in range(len(p)):
            tp = 0
            if len(g):
                j = int(np.argmax(ious[k]))
                if ious[k, j] >= 0.5 and not used[j]:
                    used[j] = True; tp = 1
            recs.append((p[k, 5], tp))
    if n_gt == 0 or not recs:
        return 0.0
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def plant_scores(items, preds, weed_pred_cls, gt_weed, gt_crop):
    """One spray score per ground-truth plant + scores of detections on bare soil.

    A plant's score is the highest weed-confidence of any weed detection
    overlapping it (IoU >= MATCH_IOU): that is the nozzle firing on it.
    Weed detections overlapping no plant are sprays wasted on soil.
    """
    rows, soil = [], []
    for img_i, ((path, gt, _), pr) in enumerate(zip(items, preds)):
        p = pr[np.isin(pr[:, 0], list(weed_pred_cls))]
        plants = gt[np.isin(gt[:, 0], [gt_weed, gt_crop])]
        ious = iou_matrix(p[:, 1:5], plants[:, 1:5])
        for j, plant in enumerate(plants):
            hit = ious[:, j] >= MATCH_IOU if len(p) else np.zeros(0, bool)
            score = float(p[hit, 5].max()) if hit.any() else 0.0
            rows.append({"image": img_i, "tray": is_tray(path),
                         "label": POSITIVE if plant[0] == gt_weed else NEGATIVE, "score": score})
        if len(p):
            free = (ious.max(axis=1) if ious.size else np.zeros(len(p))) < MATCH_IOU
            soil += [{"image": img_i, "score": float(s)} for s in p[free, 5]]
    return pd.DataFrame(rows), pd.DataFrame(soil, columns=["image", "score"])


def rates_from(df, soil, t_low, t_high):
    return outcome_rates(df.loc[df.label == POSITIVE, "score"], df.loc[df.label == NEGATIVE, "score"],
                         soil["score"], len(df), t_low, t_high)


def sweep(df, soil, costs=COSTS):
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        r = rates_from(df, soil, t, t)
        rows.append({"threshold": t,
                     "cost_per_1000": 1000 * expected_cost_per_plant(r, costs),
                     "weed_recall": r["weed"][0],
                     "crop_sprayed_rate": r["crop"][0],
                     "soil_sprays_per_plant": r["soil"][0]})
    return pd.DataFrame(rows)


def best_band(df, soil, costs=COSTS, capacity_share=None):
    """Grid search (t_low, t_high) minimising cost, optionally capped review share."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    best = None
    for i, tl in enumerate(grid):
        for th in grid[i:]:
            r = rates_from(df, soil, tl, th)
            share = review_share(r, costs)
            if capacity_share is not None and share > capacity_share:
                continue
            c = expected_cost_per_plant(r, costs)
            if best is None or c < best[0] - 1e-12:
                best = (c, float(tl), float(th), share)
    return {"cost_per_1000": 1000 * best[0], "t_low": best[1], "t_high": best[2],
            "review_share": best[3]}


def bootstrap(df, soil, t, n_images, costs=COSTS):
    """Resample test images to see how much the cost and crop-spray rate move."""
    rng = np.random.default_rng(SEED)
    by_img = {i: g for i, g in df.groupby("image")}
    soil_by = {i: g for i, g in soil.groupby("image")}
    costs_out, crop_rates = [], []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, n_images, n_images)
        d = pd.concat([by_img[i] for i in pick if i in by_img])
        s = pd.concat([soil_by[i] for i in pick if i in soil_by] or [soil.iloc[:0]])
        r = rates_from(d, s, t, t)
        costs_out.append(1000 * expected_cost_per_plant(r, costs))
        crop_rates.append(r["crop"][0])
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"cost_per_1000_ci95": q(costs_out), "crop_sprayed_rate_ci95": q(crop_rates)}


def decision_metrics(df, soil, t):
    r = rates_from(df, soil, t, t)
    w = df[df.label == POSITIVE]; c = df[df.label == NEGATIVE]
    tp = int((w.score >= t).sum()); fn = len(w) - tp
    fp = int((c.score >= t).sum()); tn = len(c) - fp
    soil_fp = int((soil.score >= t).sum())
    return {"threshold": t, "TP_weed_sprayed": tp, "FN_weed_missed": fn,
            "FP_crop_sprayed": fp, "TN_crop_left": tn, "soil_sprays": soil_fp,
            "weed_recall": tp / max(len(w), 1),
            "spray_precision": tp / max(tp + fp + soil_fp, 1),
            "crop_sprayed_rate": fp / max(len(c), 1),
            "cost_per_1000": 1000 * expected_cost_per_plant(r, COSTS)}


def ultralytics_eval(weights: Path, data_yaml: Path, names):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                               device=DEVICE, project=str(RUNS), name="test_eval",
                               exist_ok=True, plots=True, verbose=False)
    per_class = {}
    for k, ci in enumerate(m.box.ap_class_index):
        per_class[names[int(ci)]] = {"precision": float(m.box.p[k]), "recall": float(m.box.r[k]),
                                     "AP50": float(m.box.ap50[k]), "mAP50_95": float(m.box.ap[k])}
    cm = m.confusion_matrix.matrix.tolist()  # rows = predicted, cols = true (incl. background)
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "per_class": per_class, "confusion_matrix": cm,
            "confusion_labels": list(names) + ["background"],
            "confusion_note": "Ultralytics: rows predicted, cols true, conf 0.25, IoU 0.45"}


# --------------------------------------------------------------- main ---
def main():
    random.seed(SEED); np.random.seed(SEED)
    ART.mkdir(exist_ok=True)
    from ultralytics import YOLO

    data_yaml = download_dataset()
    names = yaml.safe_load(data_yaml.read_text())["names"]
    gt_weed, gt_crop = names.index(POSITIVE), names.index(NEGATIVE)
    describe(data_yaml, names)

    weights = train(data_yaml)

    log("Evaluating on the test split")
    test = read_split(data_yaml, "test")
    official = ultralytics_eval(weights, data_yaml, names)

    ft_preds = predict(YOLO(str(weights)), test)
    coco_preds = predict(YOLO(MODEL), test)  # COCO has no crop/weed: any detection = "spray"
    nothing_preds = [np.zeros((0, 6)) for _ in test]

    candidates = {
        "Finetuned yolo11n": (ft_preds, {gt_weed}, {gt_crop}),
        "COCO yolo11n zero-shot": (coco_preds, set(range(80)), set()),
        "Trivial: predict nothing": (nothing_preds, set(), set()),
    }
    baselines, scores = {}, {}
    for name, (preds, weed_set, crop_set) in candidates.items():
        df, soil = plant_scores(test, preds, weed_set or {-1}, gt_weed, gt_crop)
        sw = sweep(df, soil)
        t_best = float(sw.loc[sw.cost_per_1000.idxmin(), "threshold"])
        baselines[name] = {
            "AP50_weed": ap50(test, preds, gt_weed, weed_set) if weed_set else 0.0,
            "AP50_crop": ap50(test, preds, gt_crop, crop_set) if crop_set else 0.0,
            "best_threshold": t_best,
            **{k: v for k, v in decision_metrics(df, soil, t_best).items() if k != "threshold"},
        }
        scores[name] = (df, soil, sw)
        print(name, json.dumps(baselines[name], indent=1))

    df, soil, sw = scores["Finetuned yolo11n"]
    t_star = baselines["Finetuned yolo11n"]["best_threshold"]

    # Review band: unconstrained, then capped at what the available people can check
    cap_share = COSTS.review_capacity_per_day / COSTS.plants_per_day
    band_free = best_band(df, soil)
    band_cap = best_band(df, soil, capacity_share=cap_share)
    boot = bootstrap(df, soil, t_star, len(test))

    # Tray vs open-field photos (the shortcut risk)
    subsets = {}
    for tag, m in {"tray (IMG_)": df.tray, "open field": ~df.tray}.items():
        d = df[m]
        w, c = d[d.label == POSITIVE], d[d.label == NEGATIVE]
        subsets[tag] = {"weeds": len(w), "crops": len(c),
                        "weed_recall": float((w.score >= t_star).mean()) if len(w) else None,
                        "crop_sprayed_rate": float((c.score >= t_star).mean()) if len(c) else None}

    # Sensitivity: optimal threshold as the FN cost and base rate move
    sens = []
    for fn in [0.002, 0.005, 0.01, 0.02, 0.04]:
        for p in [0.2, 0.4, 0.6]:
            c = Costs(fn_weed_missed=fn, base_rate_weed=p)
            s = sweep(df, soil, c)
            sens.append({"fn_weed_missed": fn, "base_rate_weed": p,
                         "best_threshold": float(s.loc[s.cost_per_1000.idxmin(), "threshold"]),
                         "cost_per_1000": float(s.cost_per_1000.min())})

    r_star = rates_from(df, soil, t_star, t_star)
    r_cap = rates_from(df, soil, band_cap["t_low"], band_cap["t_high"])
    comparison = {k: 1000 * v for k, v in reference_policies(COSTS).items()}
    comparison["Model alone"] = 1000 * expected_cost_per_plant(r_star, COSTS)
    comparison["Model + review band"] = band_cap["cost_per_1000"]

    operating = {
        "positive_class": POSITIVE, "match_iou": MATCH_IOU,
        "model_alone_threshold": t_star,
        "review_band_unconstrained": band_free,
        "review_band_capacity_limited": band_cap,
        "review_capacity_share": cap_share,
        "plants_per_day": COSTS.plants_per_day,
        "reviews_per_day": band_cap["review_share"] * COSTS.plants_per_day,
        "reviewer_hours_per_day": band_cap["review_share"] * COSTS.plants_per_day
                                  * COSTS.review_seconds / 3600,
        "confusion_per_1000_model_alone": confusion_per_1000(r_star, COSTS).tolist(),
        "confusion_per_1000_with_review": confusion_per_1000(r_cap, COSTS).tolist(),
        "bootstrap": boot, "subsets": subsets, "sensitivity": sens,
        "policy_cost_per_1000": comparison, "costs": COSTS.to_dict(),
        "training": {"model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                     "seed": SEED, "patience": 25},
    }

    # ------------------------------------------------------------ save ---
    (ART / "metrics.json").write_text(json.dumps(
        {"official_test": official, "baselines": baselines}, indent=2))
    (ART / "operating_point.json").write_text(json.dumps(operating, indent=2))
    sw.to_csv(ART / "cost_sweep.csv", index=False)
    df.to_csv(ART / "plant_scores.csv", index=False)
    soil.to_csv(ART / "soil_scores.csv", index=False)
    pd.DataFrame(sens).to_csv(ART / "sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (_, _, s) in scores.items():
        ax.plot(s.threshold, s.cost_per_1000, label=name)
    ax.axvline(t_star, ls="--", c="k")
    ax.annotate(f"chosen t={t_star:.2f}", (t_star, sw.cost_per_1000.min()),
                xytext=(10, 20), textcoords="offset points")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("expected cost (€ per 1,000 plants)")
    ax.legend(); fig.tight_layout(); fig.savefig(ART / "cost_vs_threshold.png", dpi=120)

    ex = ART / "examples"; ex.mkdir(exist_ok=True)
    for f in ex.glob("*"):
        f.unlink()
    tray = [p for p, _, _ in test if is_tray(p)][:3]
    field = [p for p, _, _ in test if not is_tray(p)][:3]
    for p in tray + field:
        shutil.copy(p, ex / p.name)

    log(f"Done. Model-alone threshold {t_star:.2f}; review band "
        f"{band_cap['t_low']:.2f}-{band_cap['t_high']:.2f}. Artifacts in {ART}")


if __name__ == "__main__":
    main()
