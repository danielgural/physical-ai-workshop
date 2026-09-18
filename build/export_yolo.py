"""Export the auto-labeled frames as a YOLO detection dataset for the Nebius run.

Train/val are split by *episode* (never by frame) so validation is on scenes
the model has not seen. Near-duplicate frames are excluded from training.
Also writes the split back onto the samples as `split` so the notebooks can
show it.

    python build/export_yolo.py --out build/yolo_export
"""

import argparse
import random
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

CLASSES = ["gripper", "robot_arm", "brick", "scissors", "drawer", "cloth", "box"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="droid-frames")
    ap.add_argument("--field", default="auto_labels")
    ap.add_argument("--out", default=str(Path(__file__).with_name("yolo_export")))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-conf", type=float, default=0.3)
    args = ap.parse_args()

    ds = fo.load_dataset(args.dataset)
    episodes = sorted(ds.distinct("episode_id"))
    random.Random(51).shuffle(episodes)
    n_val = max(1, int(len(episodes) * args.val_frac))
    val_eps, train_eps = set(episodes[:n_val]), set(episodes[n_val:])

    ds.set_values("split", ["val" if e in val_eps else "train" for e in ds.values("episode_id")])

    labeled = ds.filter_labels(args.field, F("confidence") >= args.min_conf, only_matches=True)
    train = labeled.match(F("split") == "train").match_tags("near-duplicate", bool=False)
    val = labeled.match(F("split") == "val")
    print(f"episodes train/val = {len(train_eps)}/{len(val_eps)}; frames train/val = {len(train)}/{len(val)}")

    out = Path(args.out)
    for split, view in (("train", train), ("val", val)):
        view.export(export_dir=str(out), dataset_type=fo.types.YOLOv5Dataset,
                    label_field=args.field, split=split, classes=CLASSES)
    print((out / "dataset.yaml").read_text())
    # the training container expects the yaml at /data/dataset.yaml with paths relative to /data
    yaml = out / "dataset.yaml"
    text = yaml.read_text().replace(str(out), ".")
    yaml.write_text(text)
    print("->", out, "size MB:", round(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6))


if __name__ == "__main__":
    main()
