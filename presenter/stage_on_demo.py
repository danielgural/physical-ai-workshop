"""Stage the workshop datasets on demo.fiftyone.ai (FiftyOne Enterprise).

Uploads the frames (with every brain run, label field, prediction and the
evaluation) as `Droid frames`, and the 10-episode MCAP subset as
`Droid workshop episodes`. `Droid demo` (100 episodes) is already there.

Needs: the Teams SDK venv (~/Documents/development/gm/.venv), the demo API key,
and a valid `gcloud auth application-default login` for the GCS media bucket.

    ~/Documents/development/gm/.venv/bin/python presenter/stage_on_demo.py [--only frames|mcap]
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("FIFTYONE_API_URI", "https://demo-api.fiftyone.ai")
if not os.environ.get("FIFTYONE_API_KEY"):
    os.environ["FIFTYONE_API_KEY"] = open(os.path.expanduser("~/Documents/api/demo/demo.txt")).read().strip()

import fiftyone as fo  # noqa: E402
import fiftyone.core.storage as fos  # noqa: E402

GCS_ROOT = "gs://voxel51-test/physical-ai-workshop"
LOCAL_EXPORTS = Path(os.environ.get("WORKSHOP_EXPORTS", "/Users/dangural/Documents/development/physical-ai-workshop/data/exports"))
TARGETS = {
    "frames": ("droid-frames-workshop", "Droid frames"),
    "mcap": ("droid-mcap-workshop", "Droid workshop episodes"),
}


def _export_without_brain(export_dir):
    """The OSS 1.22 visualization results do not deserialize in the Enterprise
    SDK's brain version (0-d sample_ids array). Import everything else and
    recompute the brain runs on the deployment from the stored `clip` field."""
    tmp = Path(tempfile.mkdtemp(prefix="stage-"))
    for item in Path(export_dir).iterdir():
        if item.name == "brain":
            continue
        (tmp / item.name).symlink_to(item.resolve())
    return tmp


def stage(export_dir, remote_name, gcs_prefix):
    if fo.dataset_exists(remote_name):
        print(f"replacing existing {remote_name}")
        fo.delete_dataset(remote_name)
    src = _export_without_brain(export_dir) if (Path(export_dir) / "brain").exists() else Path(export_dir)
    ds = fo.Dataset.from_dir(dataset_dir=str(src), dataset_type=fo.types.FiftyOneDataset,
                             name=remote_name, persistent=True)
    ds.persistent = True  # the from_dir kwarg did not stick on Enterprise: the dataset vanished when the process exited
    print(f"{remote_name}: {len(ds)} samples imported (persistent={ds.persistent}); uploading media to {gcs_prefix}")
    fos.upload_media(ds, gcs_prefix, update_filepaths=True, overwrite=True, progress=True)
    ds.compute_metadata()

    if "clip" in ds.get_field_schema():
        import fiftyone.brain as fob

        print("recomputing brain runs from the stored clip embeddings…")
        for key in ds.list_brain_runs():  # configs came in with metadata.json but results did not
            ds.delete_brain_run(key)
        fob.compute_uniqueness(ds, embeddings="clip")
        fob.compute_similarity(ds, model="clip-vit-base32-torch", embeddings="clip", brain_key="frames_sim")
        fob.compute_visualization(ds, embeddings="clip", brain_key="frames_viz", method="umap", seed=51)

    print(ds)
    print("brain runs:", ds.list_brain_runs(), "| evaluations:", ds.list_evaluations())
    if ds.media_type == "multimodal":
        print("temporal tags:", ds.temporal_tags.count())


def recompute_only(remote_name):
    """Finish a staging whose import + upload already succeeded."""
    import fiftyone.brain as fob

    ds = fo.load_dataset(remote_name)
    for key in ds.list_brain_runs():
        ds.delete_brain_run(key)
    fob.compute_uniqueness(ds, embeddings="clip")
    fob.compute_similarity(ds, model="clip-vit-base32-torch", embeddings="clip", brain_key="frames_sim")
    fob.compute_visualization(ds, embeddings="clip", brain_key="frames_viz", method="umap", seed=51)
    print("brain runs:", ds.list_brain_runs(), "| evaluations:", ds.list_evaluations())
    print("media:", ds.first().filepath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(TARGETS))
    ap.add_argument("--recompute-runs", metavar="REMOTE_NAME", help="skip import/upload; rebuild brain runs on an existing dataset")
    args = ap.parse_args()
    if args.recompute_runs:
        return recompute_only(args.recompute_runs)
    for key, (export_name, remote_name) in TARGETS.items():
        if args.only and key != args.only:
            continue
        export_dir = LOCAL_EXPORTS / export_name
        if not export_dir.exists():
            raise SystemExit(f"missing export {export_dir}; run presenter/export_for_usb.sh {LOCAL_EXPORTS} first")
        stage(export_dir, remote_name, f"{GCS_ROOT}/{export_name}/")


if __name__ == "__main__":
    main()
