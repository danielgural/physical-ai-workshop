"""After the Nebius run: predictions on every frame, evaluation on the val
split, detector-derived temporal tags on the MCAP subset, and the numbers the
runbook quotes.

    python build/finalize.py --weights presenter/weights/best-dryrun.pt
"""

import argparse
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.core.tags import TemporalTag
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def intervals_from_predictions(frames, episodes, label, field="yolo11n_preds", min_conf=0.5, camera="ext1", gap_s=1.5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", default="droid-frames")
    ap.add_argument("--episodes", default="droid-mcap-workshop")
    args = ap.parse_args()

    frames = fo.load_dataset(args.frames)
    episodes = fo.load_dataset(args.episodes)
    model = YOLO(args.weights)

    print("predicting on all frames…")
    frames.apply_model(model, label_field="yolo11n_preds", confidence_thresh=0.25)
    print(frames.count_values("yolo11n_preds.detections.label"))

    print("evaluating on val…")
    if "eval_yolo" in frames.list_evaluations():
        frames.delete_evaluation("eval_yolo")
    val = frames.match(F("split") == "val")
    results = val.evaluate_detections("yolo11n_preds", gt_field="auto_labels", eval_key="eval_yolo", compute_mAP=True)
    results.print_report()
    mAP = results.mAP()

    print("temporal tags on the MCAP subset…")
    for label in ("brick", "gripper"):
        tag = f"{label}-visible"
        if tag in episodes.temporal_tags.count():
            episodes.temporal_tags.delete(tags=tag)
        new = intervals_from_predictions(frames, episodes, label)
        if new:
            episodes.temporal_tags.add(new)
        print(f"  {tag}: {len(new)} intervals")
    print(episodes.temporal_tags.count())

    lines = [
        "# Numbers the runbook quotes (auto-written by build/finalize.py)", "",
        f"- weights: `{args.weights}`",
        f"- frames: {len(frames)} · train/val frames {len(frames.match(F('split') == 'train'))}/{len(val)}",
        f"- auto_labels: {frames.count_values('auto_labels.detections.label')}",
        f"- yolo11n_preds: {frames.count_values('yolo11n_preds.detections.label')}",
        f"- eval_yolo mAP (val, vs auto-labels): {mAP:.3f}",
        f"- temporal tags on {args.episodes}: {episodes.temporal_tags.count()}",
    ]
    (ROOT / "presenter" / "NUMBERS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
