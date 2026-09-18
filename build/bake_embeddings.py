"""Bake the curation runs into `droid-frames`: uniqueness, CLIP embeddings,
similarity index and a UMAP visualization. Presenter-side; attendees load the
results from Hugging Face and only browse them.

    python build/bake_embeddings.py
"""

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

ds = fo.load_dataset("droid-frames")
model = foz.load_zoo_model("clip-vit-base32-torch")

if "clip" not in ds.get_field_schema() or ds.exists("clip", False).count() > 0:
    print("embeddings…")
    ds.compute_embeddings(model, embeddings_field="clip", batch_size=64, num_workers=0)

for key in ("frames_sim", "frames_viz"):
    if key in ds.list_brain_runs():
        ds.delete_brain_run(key)

print("uniqueness…")
fob.compute_uniqueness(ds, embeddings="clip")

print("similarity index…")
# model= makes the index prompt-aware: sort_by_similarity("a robot gripper …") works
fob.compute_similarity(ds, model="clip-vit-base32-torch", embeddings="clip", brain_key="frames_sim", backend="sklearn")

print("UMAP…")
fob.compute_visualization(ds, embeddings="clip", brain_key="frames_viz", method="umap", seed=51)

print("near-duplicates…")
idx = ds.load_brain_results("frames_sim")
idx.find_duplicates(thresh=0.08)
dups = idx.duplicate_ids
ds.select(dups).tag_samples("near-duplicate")
print(f"{len(dups)} near-duplicate frames tagged")
ds.save()
print(ds.list_brain_runs())
