# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**Decision.** For every plant in a field image: spray it with herbicide, leave
it, or send it to a person. Positive class = **weed** (positive means "spray").

**Who acts on it.** Today the farmer sprays the whole field with a selective
herbicide, or scouts and weeds by hand. With the model, a drone photographs the
field, the model labels each plant, an agronomist checks the uncertain ones on
a tablet, and a spot sprayer executes the resulting map with a non-selective
herbicide. Review happens before spraying, so a person can check plants
without slowing the sprayer.

**What each mistake costs** (reference crop: maize; figures are derived in
"Cost matrix and operating point"):

| Outcome | What happens | Cost per plant |
|---|---|---|
| False positive: crop sprayed | Non-selective herbicide kills the crop plant; its yield is lost | ≈ €0.020 |
| False negative: weed missed | Weed competes with nearby crop and adds seeds to the soil | ≈ €0.010 |
| True positive: weed sprayed | One spot dose of herbicide | ≈ €0.001 |
| True negative: crop left | Nothing | €0 |
| Spray on bare soil | Wasted dose | ≈ €0.001 |

**Asymmetry.** Spraying a crop is roughly **twice** as costly as missing a
weed. A killed crop plant is a certain, immediate loss. A missed weed causes a
partial, spread-out loss, and the next pass can still catch it. This pushes the
spray threshold **up**: we accept missing some weeds to protect crop plants.
The metric that matters most is therefore **how often a crop gets sprayed**,
not overall mAP.

## Dataset

- **Roboflow Universe:** workspace `object-detection`, project
  `weed-crop-aerial`, version `[fill in from artifacts/dataset_source.json]`
- **Link:** https://universe.roboflow.com/object-detection/weed-crop-aerial
- **One image:** a 640×640 top-down photo of soil with young plants. Each
  plant has a bounding box labelled `crop` or `weed`.
- **Why it serves the problem.** It labels **both** crop and weed. A
  weed-only dataset could measure missed weeds but never the expensive error,
  a crop being sprayed. It is part of Roboflow 100, a curated benchmark, so the
  labels have been reviewed.
- **Where it only resembles the problem.** The crop species is not named (a
  narrow-leaved seedling grown in planter trays), and the crop appears almost
  only in tray photos (see next section).

## What the dataset's composition implies

| Split | Images | Weed boxes | Crop boxes | Images with crop | Crop boxes in tray photos |
|---|---|---|---|---|---|
| Train | 823 | 5,188 | 278 | 70 | 275 |
| Valid | 235 | 1,558 | 47 | 17 | 43 |
| Test | 118 | 696 | 85 | 16 | 85 |

**Object size.** Weeds have a median of about 0.1% of the image area (roughly
20×20 px). Crops have a median of about 0.25% (roughly 32×32 px). These are
small objects, a known weakness of a nano detector.

**Interpretation.**

- **Imbalance.** 95% of training boxes are weeds, so the model sees few crops
  and will be weakest exactly on the class whose errors cost most.
- **The crop numbers are too thin to trust precisely.** The test set has 85
  crop boxes in only 16 images. One image with 10 crops can move crop recall by
  about 12 points, so I report bootstrap confidence intervals, resampling whole
  images.
- **The test split does not look like production.** Every test crop comes from
  tray photos (`IMG_…` files). The open-field photos contain only weeds. The
  model may learn "tray background = crop", and the crop-sprayed rate is never
  measured in open-field conditions.
- **The base rate differs.** The test set is 89% weeds. A planted field has
  far more crop plants, so the cost analysis uses an assumed production base
  rate, not the dataset's.
- **No leakage.** No source image appears in more than one split.

**What I cannot conclude.** That the crop-sprayed rate holds in an open field;
results only hold for tray-like scenes. Precise crop precision and recall.

## Finetuning and baselines

**Model and task.** `yolo11n`, **object detection**. Detection rather than
classification because one image contains many plants and the sprayer needs a
location per plant. Nano because a spot sprayer or drone needs fast
inference.

**Training.** `[where: e.g. Google Colab, T4 GPU]`, from COCO-pretrained
weights, up to 100 epochs, early stopping with patience 25, image size 640,
batch 16, seed 0.

**Results on the test split** (each at its own cost-minimising threshold):

| Model | AP50 weed | AP50 crop | Weed recall | Crops sprayed | € / 1,000 plants |
|---|---|---|---|---|---|
| Finetuned yolo11n | | | | | |
| COCO yolo11n zero-shot | | | | | |
| Trivial: predict nothing | 0 | 0 | 0 | 0 | |

*Fill in from artifacts/metrics.json. COCO has no crop or weed class, so every
COCO detection is treated as "spray". Comment: what changed, and whether the
gap is as large as expected.*

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| weed | | | | |
| crop | | | | |
| all (headline) | | | | |

*Fill in from artifacts/metrics.json, plus the confusion matrix. Points to cover:*

- *Weed carries the headline number (89% of test boxes) and hides crop weakness.*
- *Crop called weed is the costly confusion: how many?*
- *Tiny weeds missed (confused with background).*
- *Tray photos vs open-field photos at the chosen threshold (operating_point.json, "subsets").*

## Cost matrix and operating point

**Where each figure comes from** (assumptions, reference crop maize):

- **Value of one crop plant.** Maize yields about 10 t/ha at about €200/t, so
  roughly €2,000/ha, over about 80,000 plants/ha, giving about €0.025 per
  plant.
- **FP, crop sprayed: €0.020.** The plant's value minus some compensation, as
  neighbouring plants grow into the gap, plus one wasted dose.
- **FN, weed missed: €0.010.** Yield lost to competition plus future seed
  bank. This is the least certain figure.
- **TP, weed sprayed: €0.001.** One spot dose. **Soil spray: €0.001.**
- **Base rate: 40% weeds.** A moderately infested field after planting,
  instead of the dataset's 89%.

**Sweep.** Every threshold from 0.01 to 0.99 is costed per 1,000 plants. The
curve is in `artifacts/cost_vs_threshold.png`.

**Chosen operating point:** `t = [fill in]`, at `€[ ]` per 1,000 plants
(bootstrap 95% CI `[ ]`).

**Assumption that would most change the answer:** the FN/FP cost ratio. If a
missed weed cost as much as a sprayed crop, the threshold would drop to
`[ ]` (artifacts/sensitivity.csv). The base rate matters next: with fewer
weeds, crop protection dominates and the threshold rises.

## What a person still has to check

**Thresholds:** below `t_low = [ ]` don't spray; above `t_high = [ ]` spray;
in between, the agronomist checks on a tablet.

**Volume.** 20 ha surveyed per day × 80,000 crop plants/ha ÷ 60% crop share
gives about **2.7 million plants/day**. One check takes about 2.5 s, so one
reviewer working 8 h can do about 11,500 checks, which is **0.43% of
plants**. The review band is capped at that share.

**Is review worth it?** One check costs about €0.014 (2.5 s at €20/h). That
is more than a missed weed costs, so reviewing weeds never pays; reviewing
only pays for plants likely to be crops above the spray threshold. *Comment on
the band the sweep found and whether one person is realistic.*

| Policy | € per 1,000 plants |
|---|---|
| Do nothing (never spray) | 4.00 |
| Person checks every plant | 14.29 |
| Model alone | |
| Model + review band | |

*Spraying every plant costs €12.40, a useful extra reference. Fill in from
operating_point.json, "policy_cost_per_1000".*

## Limitations & next steps

- **Crop only in tray photos.** Collect open-field images that contain the
  crop, so the crop-sprayed rate is measured where the sprayer works.
- **Thin crop test set** (85 boxes, 16 images). Add data, or use
  cross-validation, and keep reporting bootstrap intervals.
- **Small objects.** Try image size 960, tiling, or `yolo11s`.
- **Assumed costs.** Validate plant value, weed damage and review time with an
  agronomist, and fix the actual crop species.
- **Per-plant review is expensive.** Test reviewing a whole image tile at once,
  which would lower the cost per plant checked.

## Generative AI use disclosure

*Describe what you used and how.*
