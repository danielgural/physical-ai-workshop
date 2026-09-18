"""Rebuild frames_sim as a prompt-aware CLIP index (so text queries work)."""
import fiftyone as fo
import fiftyone.brain as fob

ds = fo.load_dataset("droid-frames")
if "frames_sim" in ds.list_brain_runs():
    ds.delete_brain_run("frames_sim")
fob.compute_similarity(ds, model="clip-vit-base32-torch", embeddings="clip", brain_key="frames_sim", backend="sklearn")
v = ds.sort_by_similarity("a robot gripper holding a blue brick", k=5, brain_key="frames_sim")
print([ (s.episode_id[-12:], s.camera) for s in v])
