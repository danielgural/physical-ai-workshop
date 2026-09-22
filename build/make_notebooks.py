"""Generate the attendee notebooks 01–06 for physical-ai-workshop."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    nbf.write(n, OUT / name)
    print("wrote", name)


md, code = "md", "code"

# ---------------------------------------------------------------- 01
nb("01_explore_mcap.ipynb", [
(md, """# 01 · Explore a robot recording

**Agenda: 10–25 min.** A DROID teleop episode is one `.mcap` file: three cameras, depth, fused point clouds, 3D cuboids, joint state, gripper state, and an episode summary — on one clock. In FiftyOne ≥ 1.21 a sample whose filepath is an `.mcap` **is** the episode. No conversion, no ROS.

You have 10 episodes locally (`droid-mcap-workshop`). The presenter has 100 on demo.fiftyone.ai. Same workflow."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

episodes = load("droid-mcap-workshop")
print(episodes.media_type, len(episodes), "episodes")
episodes"""),
(code, """session = fo.launch_app(episodes)"""),
(md, """## In the App

1. **Grid** — every tile is an episode with a playing preview. Use the stream selector to switch the preview camera.
2. **Open an episode** — arrange the tiles: wrist camera, an external camera, the 3D view. Scrub. Everything moves on one clock.
3. **Click a cuboid** in the 3D tile → label, entity id, topic.
4. **Timeline tracks** — expand them. `grasp` / `release` intervals are *temporal tags*: labels on a time range, not a frame.
5. **Sidebar** — filter by `task_type`, `success`, sort by `duration_s`.

### Episodes are queryable like any dataset"""),
(code, """print(episodes.count_values("task_type"))
print(episodes.count_values("success"))

# Suspiciously short "successful" demonstrations
short = episodes.match(F("duration_s") < 4).sort_by("duration_s")
for s in short.select_fields(["episode_id", "duration_s", "success", "current_task", "gripper_open_frac"]):
    print(f"{s.duration_s:5.1f}s  success={s.success!s:5}  gripper_open_frac={s.gripper_open_frac:.2f}  {s.current_task.splitlines()[0][:60]}")"""),
(md, """The 1.4 s episode: the gripper never closed (`gripper_open_frac == 1.0`), marked failed. The 2.8 s one is marked a *success* — worth a look. Open both."""),
(code, """session.view = short"""),
(md, """### Temporal tags: labels on a time range

The dataset ships with `grasp` and `release` intervals. Query them:"""),
(code, """print(episodes.temporal_tags.count())

grasping = episodes.match_temporal_tags(tags="grasp")
print(len(grasping), "episodes contain a grasp interval")
session.view = grasping"""),
(md, """**Try it:** open an episode, press **Shift+T** for tag mode, drag an interval on the timeline and name it `arm-stall`. Then:"""),
(code, """# after tagging in the App
print(episodes.temporal_tags.count())
print(len(episodes.match_temporal_tags(tags="arm-stall")), "episodes with an arm-stall")"""),
(md, """### What just happened

- One file per episode, browsable and queryable without a conversion pipeline.
- Curation questions ("which demos are degenerate?") are field queries, not scripts.
- Quality lives at the *episode* and *interval* level — the temporal tags are the unit we will come back to in notebook 06.

**Next:** to train a detector we need images. Notebook 02 pulls frames out of these recordings."""),
])

# ---------------------------------------------------------------- 02
nb("02_curate_frames.ipynb", [
(md, """# 02 · Curate: frames out of the recording

**Agenda: 25–40 min.** Detectors train on images. The RGB cameras inside each MCAP are H.264 streams, so `build/extract_frames.py` decodes them through ffmpeg at 1 fps, resizes to 640 px, and writes a FiftyOne image dataset where every frame remembers its **episode, camera, timestamp**, and the robot state at that instant (gripper open/closed, joint speed).

You are loading the result: 5,172 frames from the 75 episodes that carry video."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

frames = load("droid-frames-workshop")
print(len(frames), "frames from", len(frames.distinct("episode_id")), "episodes")
print(frames.count_values("camera"))
frames"""),
(code, """session = fo.launch_app(frames)"""),
(md, """### The extraction, in one cell (reference — do not run on the full set now)

```python
# build/extract_frames.py, the heart of it
proc = subprocess.Popen(["ffmpeg", "-f", "h264", "-i", "pipe:0",
                         "-vf", f"fps={fps},scale={width}:-2", "-q:v", "3", out_pattern], stdin=subprocess.PIPE)
for _, channel, message, proto in reader.iter_decoded_messages(topics=["/camera/wrist/image_rgb"]):
    proc.stdin.write(proto.data)          # raw H.264 NAL units → ffmpeg
```

Each output frame becomes an `fo.Sample` with `episode_id`, `camera`, `timestamp_ns`, `gripper_open`, `joint_speed_norm`. That link back to the recording is what makes notebook 06 possible."""),
(md, """## Filter with robot state, not just pixels

Frames where the gripper is closed are the ones where something is being held — the interesting ones for a gripper/object detector."""),
(code, """closed = frames.match(F("gripper_open") == False)  # noqa: E712
print(len(closed), "frames with the gripper closed")

moving = frames.match(F("joint_speed_norm") > 0.8)
print(len(moving), "frames while the arm is moving fast (motion blur candidates)")

session.view = closed.match(F("camera") == "wrist")"""),
(md, """## Uniqueness and near-duplicates

1 fps on a slow-moving arm produces many near-identical frames. `compute_uniqueness` (precomputed) scores how unusual each frame is relative to the rest; frames the similarity index flagged as near-duplicates carry the tag `near-duplicate`."""),
(code, """print(frames.bounds("uniqueness"))
print(frames.count_sample_tags())

# Least unique frames: the bench with nothing happening
session.view = frames.sort_by("uniqueness")"""),
(code, """# Most unique: odd viewpoints, hands in frame, dropped objects
session.view = frames.sort_by("uniqueness", reverse=True)"""),
(md, """## Build the curation view

Keep external-camera frames, drop near-duplicates, prefer the ones where the gripper is closed or the arm is moving. Save it — a saved view is a reusable query, not a copy of the data."""),
(code, """curated = (frames
           .match_tags("near-duplicate", bool=False)
           .match(F("camera") != "wrist")
           .match((F("gripper_open") == False) | (F("joint_speed_norm") > 0.3)))  # noqa: E712
print(len(curated), "frames in the curated view")

frames.save_view("curated_train", curated, overwrite=True)
session.view = curated"""),
(md, """**Optional (≈1 min on CPU):** recompute uniqueness yourself on a 300-frame slice to see how it works."""),
(code, """# OPTIONAL — ~1 min on CPU
# import fiftyone.brain as fob
# slice_ = frames.take(300, seed=51)
# fob.compute_uniqueness(slice_, uniqueness_field="uniqueness_mine")
# session.view = slice_.sort_by("uniqueness_mine", reverse=True)"""),
(md, """**Next:** the curated view is what we embed, label and train on. Notebook 03 looks at it through embeddings."""),
])

# ---------------------------------------------------------------- 03
nb("03_embeddings.ipynb", [
(md, """# 03 · Embeddings, similarity, visualization

**Agenda: 40–55 min.** Every frame has a CLIP embedding (`clip`), a similarity index (`frames_sim`) and a 2-D UMAP layout (`frames_viz`). The presenter computes these on Nebius for the full deployment; here they are precomputed so the workflow stays interactive on a laptop."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

frames = load("droid-frames-workshop")
print(frames.list_brain_runs())
session = fo.launch_app(frames)"""),
(md, """## Embeddings panel

Open the **Embeddings** panel (the `+` next to Samples), choose `frames_viz`.

1. **Color by `camera`** — three clusters: wrist, ext1, ext2. The wrist camera lives in its own world.
2. **Color by `task_type`** — within each camera cluster, fold-cloth / clump / brick tasks separate.
3. **Color by `gripper_open`** — does the embedding know when something is being held?
4. **Lasso** a small island far from everything → the grid shows what it is. Usually a human in frame, a dropped object, or a camera knocked out of place."""),
(md, """## Similarity search

Pick one frame in the grid, click the image-search icon (or use the code below) → the grid re-sorts by similarity. This is how you find *more of a failure* once you have seen one."""),
(code, """# The frame with the highest uniqueness: find its neighbours
odd = frames.sort_by("uniqueness", reverse=True).first()
similar = frames.sort_by_similarity(odd.id, k=50, brain_key="frames_sim")
session.view = similar"""),
(md, """## Text search against the same index

The index was built with CLIP, so it also takes text prompts."""),
(code, """for prompt in ["a robot gripper holding a blue brick", "a person's hand", "an open drawer"]:
    view = frames.sort_by_similarity(prompt, k=24, brain_key="frames_sim")
    print(prompt, "→", view.first().episode_id, view.first().camera)

session.view = frames.sort_by_similarity("a robot gripper holding a blue brick", k=48, brain_key="frames_sim")"""),
(md, """## Optional: compute embeddings yourself (≈1–2 min on CPU for 200 frames)"""),
(code, """# OPTIONAL — 200 frames, CPU
# import fiftyone.brain as fob
# import fiftyone.zoo as foz
# slice_ = frames.take(200, seed=51)
# model = foz.load_zoo_model("clip-vit-base32-torch")
# slice_.compute_embeddings(model, embeddings_field="clip_mine", num_workers=0)
# fob.compute_visualization(slice_, embeddings="clip_mine", brain_key="mine_viz", method="umap")
# session.view = slice_"""),
(md, """### Where this ran for the presenter

The same `compute_embeddings` call, on the 100-episode deployment, runs as a **Nebius Serverless AI Job**: a GPU container that reads the frames from object storage and writes vectors back. The interactive part — lasso, sort, tag — stays on the laptop.

**Next:** we have frames worth labeling. Notebook 04 labels them without a human drawing a box."""),
])

# ---------------------------------------------------------------- 04
nb("04_autolabel.ipynb", [
(md, """# 04 · Auto-label with open-vocabulary detection

**Agenda: 55–75 min.** No one is going to hand-draw boxes on 5,000 robot frames. An open-vocabulary detector (Grounding DINO, prompted with plain words) does the first pass; humans review the queue.

Prompts used: `robot gripper . robot arm . blue brick . scissors . drawer . cloth . cardboard box`, folded onto 7 labels. The result is on every frame as `auto_labels`."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

frames = load("droid-frames-workshop")
print(frames.count_values("auto_labels.detections.label"))
session = fo.launch_app(frames)"""),
(md, """## Look before you trust

Sort by number of detections, look at the empty frames and the crowded ones. Confidence is stored per box — filter it in the sidebar."""),
(code, """n_dets = F("auto_labels.detections").length()
print("frames with no labels:", len(frames.match(n_dets == 0)))
print("frames with 8+ labels:", len(frames.match(n_dets >= 8)))

# Gripper boxes only, low confidence first: the ones a reviewer should check
gripper_low = (frames.filter_labels("auto_labels", (F("label") == "gripper") & (F("confidence") < 0.4))
                     .sort_by(F("auto_labels.detections").length(), reverse=True))
session.view = gripper_low"""),
(md, """## Sanity check the labels against robot state

The gripper is visible in almost every wrist-camera frame. If the labeler misses it there, that is a labeler problem, not a data problem."""),
(code, """wrist = frames.match(F("camera") == "wrist")
has_gripper = wrist.filter_labels("auto_labels", F("label") == "gripper", only_matches=True)
print(f"wrist frames: {len(wrist)}, with a gripper box: {len(has_gripper)} ({100*len(has_gripper)/len(wrist):.0f}%)")

# Brick boxes should live on the brick-in-drawer episodes. Where does the labeler actually put them?
brick = frames.filter_labels("auto_labels", F("label") == "brick", only_matches=True)
print(brick.count_values("task_type"))
session.view = brick.match(F("task_type") == "clump-unclump")"""),
(md, """Most "brick" boxes land on **clump-unclump** episodes — there is no brick there. The prompt said *blue brick*; the labeler grabbed blue cloth and plush toys. That is a prompt problem, and it is exactly the kind of thing that only shows up when labels are checked against metadata you already have (the task string). Fix the prompt, or drop `brick` from the training classes for those tasks."""),
(md, """## Optional: run Grounding DINO yourself on 20 frames (≈30 s on Apple Silicon, ~2 min CPU)"""),
(code, """# OPTIONAL
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# mid = "IDEA-Research/grounding-dino-tiny"
# proc = AutoProcessor.from_pretrained(mid); model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
# text = "robot gripper . blue brick . scissors ."
# for s in frames.take(20, seed=7).iter_samples(autosave=True):
#     im = Image.open(s.filepath).convert("RGB"); w, h = im.size
#     inp = proc(images=im, text=text, return_tensors="pt")
#     with torch.no_grad(): out = model(**inp)
#     r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(h, w)])[0]
#     s["my_labels"] = fo.Detections(detections=[
#         fo.Detection(label=str(l), confidence=float(c), bounding_box=[b[0]/w, b[1]/h, (b[2]-b[0])/w, (b[3]-b[1])/h])
#         for l, c, b in zip(r["text_labels"] if "text_labels" in r else r["labels"], r["scores"], r["boxes"].tolist())])
# session.view = frames.exists("my_labels")"""),
(md, """### The training set

`split` was assigned **by episode** (train 60 / val 15 episodes) so validation frames come from scenes the model never saw. Near-duplicates are excluded from training. This is what went to Nebius in notebook 05."""),
(code, """print(frames.count_values("split"))
train = frames.match(F("split") == "train").match_tags("near-duplicate", bool=False)
print(len(train), "training frames")"""),
])

# ---------------------------------------------------------------- 05
nb("05_train_eval.ipynb", [
(md, """# 05 · Train YOLO11n and evaluate

**Agenda: 75–95 min.** The presenter exports the auto-labeled frames to YOLO format, stages them on Nebius Object Storage, and submits a **Nebius Serverless AI Job** from a notebook (`presenter/train_on_nebius.ipynb`): an L40S GPU container fine-tunes `yolo11n.pt` for 15 epochs and drops `best.pt` back in the bucket.

That model's predictions are on your frames as `yolo11n_preds`, and the evaluation against the auto-labels is the run `eval_yolo`."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

frames = load("droid-frames-workshop")
print(frames.list_evaluations())
session = fo.launch_app(frames)"""),
(md, """## Model Evaluation panel

Open the **Model Evaluation** panel → `eval_yolo`. mAP per class, confusion matrix, and every cell of the matrix is clickable: click *gripper predicted as robot_arm* and the grid shows those frames.

Or in code:"""),
(code, """results = frames.load_evaluation_results("eval_yolo")
results.print_report()"""),
(code, """# Validation frames only, worst first (most false positives)
val = frames.match(F("split") == "val")
session.view = val.sort_by("eval_yolo_fp", reverse=True)"""),
(code, """# Missed grippers (false negatives) — what does the detector not see?
fn = val.filter_labels("auto_labels", (F("eval_yolo") == "fn") & (F("label") == "gripper"), only_matches=True)
print(len(fn), "val frames with a missed gripper")
session.view = fn"""),
(md, """## The job spec, for reference

```python
from presenter.nebius_job import NebiusJob
job = NebiusJob()                                     # Nebius IAM token + Object Storage key
job.upload_dir("build/yolo_export", "runs/stuttgart/input")
job_id = job.submit("stuttgart", "runs/stuttgart", model="yolo11n.pt", epochs=15,
                    platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb")
job.wait(job_id)                                      # PROVISIONING → RUNNING → COMPLETED
job.download("runs/stuttgart/output/best.pt", "presenter/weights/best.pt")
```

The container is public (`ghcr.io/danielgural/fiftyone-yolo-train`) and only needs `INPUT_S3_URI`, `OUTPUT_S3_URI`, `MODEL`, `HYPERPARAMS_JSON`."""),
(md, """## Optional: a 1-epoch fine-tune on your laptop (≈3–5 min CPU, 200 frames)

Same code as the GPU job, tiny scale. `ultralytics` is in `requirements.txt`."""),
(code, """# OPTIONAL
# from ultralytics import YOLO
# small = frames.match(F("split") == "train").match_tags("near-duplicate", bool=False).take(200, seed=51)
# small.export(export_dir="/tmp/yolo_small", dataset_type=fo.types.YOLOv5Dataset, label_field="auto_labels", split="train",
#              classes=["gripper", "robot_arm", "brick", "scissors", "drawer", "cloth", "box"])
# frames.match(F("split") == "val").take(60, seed=51).export(export_dir="/tmp/yolo_small", dataset_type=fo.types.YOLOv5Dataset,
#              label_field="auto_labels", split="val", classes=["gripper", "robot_arm", "brick", "scissors", "drawer", "cloth", "box"])
# model = YOLO("yolo11n.pt")
# model.train(data="/tmp/yolo_small/dataset.yaml", epochs=1, imgsz=416, batch=16, device="cpu", workers=0, plots=False)
# frames.match(F("split") == "val").take(60, seed=51).apply_model(model, label_field="my_preds")
# session.view = frames.exists("my_preds")"""),
(md, """**Next:** predictions on frames are useful. Predictions *on the recording's timeline* are what the robotics team actually wants. Notebook 06."""),
])

# ---------------------------------------------------------------- 06
nb("06_close_the_loop.ipynb", [
(md, """# 06 · Close the loop: back to the MCAP timeline

**Agenda: 95–105 min.** Every frame knows its `episode_id` and `timestamp_ns`. So detector output on frames can be turned back into **temporal tags** on the episode — intervals where the gripper, or a brick, is visible — and queried exactly like the grasp/release tags we started with."""),
(code, """import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh

def load(name):
    # Local copy if download_data.py already ran, otherwise pull it from Hugging Face now
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    print(f"{name} not found locally; downloading from Hugging Face (one time)…")
    return fouh.load_from_hub(f"dgural/{name}", name=name, persistent=True)

episodes = load("droid-mcap-workshop")
frames = load("droid-frames-workshop")
print(episodes.temporal_tags.count())"""),
(md, """The dataset already carries `brick-visible` and `gripper-visible` tags derived from the Nebius-trained YOLO11n. Here is how they were built, and you can rebuild them for any label:"""),
(code, """from fiftyone.core.tags import TemporalTag

def intervals_from_predictions(frames, episodes, label, field="yolo11n_preds", min_conf=0.5, camera="ext1", gap_s=1.5):
    \"\"\"Detections on 1-fps frames -> merged [start, end] intervals per episode (ns, episode-relative).\"\"\"
    tags = []
    ep_ids = {s.episode_id: s.id for s in episodes.select_fields("episode_id")}
    hits = (frames.match(F("camera") == camera)
                  .filter_labels(field, (F("label") == label) & (F("confidence") >= min_conf), only_matches=True)
                  .match(F("episode_id").is_in(list(ep_ids))))
    by_ep = {}
    for ep, t in zip(*hits.values(["episode_id", "t_rel_s"])):
        by_ep.setdefault(ep, []).append(t)
    for ep, ts in by_ep.items():
        ts.sort()
        start = prev = ts[0]
        for t in ts[1:] + [None]:
            if t is None or t - prev > gap_s:
                tags.append(TemporalTag(sample_id=ep_ids[ep], start=int(start * 1e9), end=int((prev + 1.0) * 1e9),
                                        tag=f"{label}-visible", index_type=2))
                if t is not None:
                    start = t
            if t is not None:
                prev = t
    return tags

new_tags = intervals_from_predictions(frames, episodes, "brick")
print(len(new_tags), "brick-visible intervals across", len({t.sample_id for t in new_tags}), "episodes")"""),
(code, """# Write them (idempotent: clear the tag first)
episodes.temporal_tags.delete(tags="brick-visible-mine") if "brick-visible-mine" in episodes.temporal_tags.count() else None
for t in new_tags:
    t.tag = "brick-visible-mine"
episodes.temporal_tags.add(new_tags)
print(episodes.temporal_tags.count())"""),
(code, """session = fo.launch_app(episodes)
session.view = episodes.match_temporal_tags(tags="brick-visible")"""),
(md, """Open an episode: the timeline now has `grasp`, `release`, and `brick-visible` rows. Scrub to a `brick-visible` interval — the external camera should show the brick.

## Rank episodes by what the detector saw"""),
(code, """# Episodes where the model is least confident about the gripper: candidates for more data or review
conf = (frames.match(F("camera") == "ext1")
              .filter_labels("yolo11n_preds", F("label") == "gripper", only_matches=True)
              .match(F("episode_id").is_in(episodes.distinct("episode_id"))))
per_ep = {}
for ep, dets in zip(*conf.values(["episode_id", "yolo11n_preds.detections.confidence"])):
    per_ep.setdefault(ep, []).extend(dets)
for ep, cs in sorted(per_ep.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    print(f"{sum(cs)/len(cs):.2f}  {len(cs):3d} gripper boxes  {ep}")"""),
(md, """### The loop

**see** the recording → **curate** frames with robot state → **embed** and find the odd ones → **auto-label** → **train** on Nebius → **evaluate** → **tag the recording** with what the model found → the next collection run is curated with these saved views and tags.

Same tools, same loop, at 100 episodes on a laptop or millions in a deployment.

- Try it on your own recordings: drop any `.mcap` into the **MCAP Explorer** panel.
- Hosted FiftyOne: [app.voxel51.com](https://app.voxel51.com)
- GPUs for the loop: [Nebius Builder Program](https://link.voxel51.com/nebius)"""),
])
