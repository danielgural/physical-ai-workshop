"""Build `droid-mcap-workshop`: the 10-episode multimodal MCAP subset attendees
download. Copies the MCAP files, keeps every episode-level field and the
grasp/release temporal tags, drops the 100-episode brain runs (a UMAP of ten
points is not a demo).

    python build/build_mcap_subset.py
"""

import json
import shutil
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.core.tags import TemporalTag

NAME = "droid-mcap-workshop"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mcap-subset"


def main():
    episodes = json.loads((ROOT / "build" / "episodes.json").read_text())
    ids = [e["episode_id"] for e in episodes]
    droid = fo.load_dataset("droid")
    view = droid.match(F("episode_id").is_in(ids))
    assert len(view) == len(ids), (len(view), len(ids))

    if fo.dataset_exists(NAME):
        fo.delete_dataset(NAME)
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = fo.Dataset(NAME, persistent=True)
    dataset.tags = ["robotics", "mcap", "multimodal", "droid", "workshop"]
    dataset.info = droid.info | {
        "source": "DROID (droid-dataset.github.io), CC-BY-4.0",
        "note": "10-episode subset of dgural/droid-mcap-demo for the Nebius x Voxel51 Physical AI workshop",
    }

    old_to_new = {}
    samples = []
    for s in view:
        dst = OUT / Path(s.filepath).name
        if not dst.exists():
            shutil.copy2(s.filepath, dst)
        d = s.to_dict()
        for k in ("_id", "id", "created_at", "last_modified_at", "metadata", "filepath",
                  "embeddings", "cam_embedding", "depth_embedding", "points_embedding"):
            d.pop(k, None)
        new = fo.Sample(filepath=str(dst), **d)
        samples.append((s.id, new))
    dataset.add_samples([n for _, n in samples])
    for (old_id, new), nid in zip(samples, dataset.values("id")):
        old_to_new[old_id] = nid

    # Temporal tags (grasp / release intervals) come along, re-keyed to the new sample ids
    carried = []
    for tag in droid.temporal_tags.values():
        if tag.sample_id in old_to_new:
            carried.append(TemporalTag(
                sample_id=old_to_new[tag.sample_id], start=tag.start, end=tag.end,
                tag=tag.tag, index_type=tag.index_type, anchor=tag.anchor))
    if carried:
        dataset.temporal_tags.add(carried)

    dataset.compute_metadata()
    dataset.save()
    print(dataset)
    print("media type:", dataset.media_type, "| temporal tags:", dataset.temporal_tags.count())
    print("subset size MB:", round(sum(p.stat().st_size for p in OUT.iterdir()) / 1e6))


if __name__ == "__main__":
    main()
