"""Stage the frames dataset on demo.fiftyone.ai as `Droid workshop frames`, one step per process.

    V=~/Documents/development/gm/.venv/bin/python
    for s in import check upload check brain check; do $V presenter/stage_frames_on_demo.py $s; done

Why steps: the OSS 1.22 brain result files do not deserialize in the Enterprise SDK
(0-d sample_ids array), so `brain/` is skipped on import and the runs are recomputed
from the stored `clip` field. A single-process run of the same flow once left no dataset
behind on the deployment; running each step in its own process and checking in between
is the version that has been verified to stick. Needs `gcloud auth application-default login`.
"""
import os, sys, subprocess, tempfile
from pathlib import Path
os.environ["FIFTYONE_API_URI"] = "https://demo-api.fiftyone.ai"
os.environ["FIFTYONE_API_KEY"] = open(os.path.expanduser("~/Documents/api/demo/demo.txt")).read().strip()
import fiftyone as fo
NAME = "Droid workshop frames"
EXPORT = Path("/Users/dangural/Documents/development/physical-ai-workshop/data/exports/droid-frames-workshop")
GCS = "gs://voxel51-test/physical-ai-workshop/droid-frames-workshop/"
step = sys.argv[1]
if step == "check":
    print("CHECK exists:", fo.dataset_exists(NAME), "|", (len(fo.load_dataset(NAME)) if fo.dataset_exists(NAME) else "-"), "|", fo.load_dataset(NAME).persistent if fo.dataset_exists(NAME) else "-")
elif step == "import":
    for n in (NAME, "Droid frames"):
        if fo.dataset_exists(n): fo.delete_dataset(n); print("deleted", n)
    tmp = Path(tempfile.mkdtemp(prefix="stage-"))
    for item in EXPORT.iterdir():
        if item.name != "brain": (tmp / item.name).symlink_to(item.resolve())
    ds = fo.Dataset.from_dir(dataset_dir=str(tmp), dataset_type=fo.types.FiftyOneDataset, name=NAME, persistent=True)
    ds.persistent = True
    print("IMPORT", ds.name, len(ds), ds.persistent, ds.list_brain_runs(), ds.list_evaluations())
elif step == "upload":
    import fiftyone.core.storage as fos
    ds = fo.load_dataset(NAME)
    fos.upload_media(ds, GCS, update_filepaths=True, overwrite=True, progress=False)
    ds.compute_metadata()
    print("UPLOAD", len(set(ds.values("filepath"))), "distinct paths;", ds.first().filepath)
elif step == "brain":
    import fiftyone.brain as fob
    ds = fo.load_dataset(NAME)
    for key in ds.list_brain_runs(): ds.delete_brain_run(key)
    fob.compute_uniqueness(ds, embeddings="clip")
    fob.compute_similarity(ds, model="clip-vit-base32-torch", embeddings="clip", brain_key="frames_sim")
    fob.compute_visualization(ds, embeddings="clip", brain_key="frames_viz", method="umap", seed=51)
    print("BRAIN", ds.list_brain_runs())
