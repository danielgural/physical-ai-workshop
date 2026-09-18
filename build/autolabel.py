"""Auto-label the extracted frames with an open-vocabulary detector.

Grounding DINO (tiny) prompted with the objects that actually appear on the
AUTOLab bench. Writes `auto_labels` (fo.Detections) on every frame of
`droid-frames`. These become the training targets for the YOLO11n fine-tune.

    python build/autolabel.py [--limit N]
"""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import fiftyone as fo

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPTS = ["robot gripper", "robot arm", "blue brick", "scissors", "drawer", "cloth", "cardboard box"]
# Grounding DINO returns free-text phrases; fold them onto a fixed label set.
CANON = [
    ("gripper", "gripper"), ("arm", "robot_arm"), ("brick", "brick"), ("blue", "brick"),
    ("scissors", "scissors"), ("drawer", "drawer"), ("cloth", "cloth"), ("box", "box"),
]
THRESH, TEXT_THRESH = 0.3, 0.25


def canon(phrase):
    for needle, label in CANON:
        if needle in phrase:
            return label
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="droid-frames")
    ap.add_argument("--field", default="auto_labels")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(dev).eval()
    text = ". ".join(PROMPTS) + "."

    ds = fo.load_dataset(args.dataset)
    view = ds.exists(args.field, False) if args.field in ds.get_field_schema() else ds
    if args.limit:
        view = view.limit(args.limit)
    print(f"{len(view)} frames to label on {dev}")

    for sample in view.iter_samples(autosave=True, batch_size=32, progress=True):
        im = Image.open(sample.filepath).convert("RGB")
        w, h = im.size
        inp = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=THRESH, text_threshold=TEXT_THRESH,
            target_sizes=[(h, w)])[0]
        phrases = res["text_labels"] if "text_labels" in res else res["labels"]
        dets = []
        for phrase, score, box in zip(phrases, res["scores"], res["boxes"]):
            label = canon(str(phrase))
            if label is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            dets.append(fo.Detection(
                label=label,
                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                confidence=float(score),
            ))
        sample[args.field] = fo.Detections(detections=dets)

    print(ds.count_values(f"{args.field}.detections.label"))


if __name__ == "__main__":
    main()
