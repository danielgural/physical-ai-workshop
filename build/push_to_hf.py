"""Push the two workshop datasets to Hugging Face (public, CC-BY-4.0).

    python build/push_to_hf.py mcap      # dgural/droid-mcap-workshop
    python build/push_to_hf.py frames    # dgural/droid-frames-workshop
"""

import sys

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

SPECS = {
    "mcap": dict(
        dataset="droid-mcap-workshop",
        repo="droid-mcap-workshop",
        preview="slides/assets/episode-viewer-3d.png",
        tags=["robotics", "mcap", "multimodal", "droid", "manipulation", "workshop"],
        description=(
            "10 DROID robot-arm teleop episodes as multimodal MCAP recordings for the Nebius x Voxel51 "
            "Physical AI workshop: 3 synchronized cameras, fused point clouds, 3D cuboid tracks, joint and "
            "gripper state, grasp/release temporal tags, plus detector-derived 'brick-visible' / "
            "'gripper-visible' intervals and episode-level task/outcome/motion fields. One FiftyOne sample per "
            "episode; requires fiftyone>=1.21.0 (multimodal media type). Subset of dgural/droid-mcap-demo. "
            "Derived from DROID (https://droid-dataset.github.io), CC-BY-4.0."
        ),
    ),
    "frames": dict(
        dataset="droid-frames",
        repo="droid-frames-workshop",
        preview="slides/assets/frame-ext1.jpg",
        tags=["robotics", "droid", "manipulation", "object-detection", "workshop"],
        description=(
            "5,172 RGB frames (1 fps, 640 px) decoded from the camera streams of 75 DROID robot-arm episodes, "
            "for the Nebius x Voxel51 Physical AI workshop. Every frame keeps its episode_id, camera and "
            "timestamp plus the robot state at that instant (gripper_open, joint_speed_norm, ee_linear_speed), "
            "and carries: CLIP embeddings with similarity (frames_sim) and UMAP (frames_viz) brain runs, "
            "uniqueness, near-duplicate tags, an episode-level train/val split, Grounding DINO open-vocabulary "
            "auto-labels (auto_labels), YOLO11n predictions from a Nebius Serverless AI Jobs fine-tune "
            "(yolo11n_preds) and the evaluation run eval_yolo. Derived from DROID "
            "(https://droid-dataset.github.io), CC-BY-4.0."
        ),
    ),
}


def main(which):
    spec = SPECS[which]
    ds = fo.load_dataset(spec["dataset"])
    print(f"pushing {spec['dataset']} ({len(ds)} samples) -> dgural/{spec['repo']}")
    fouh.push_to_hub(
        ds, spec["repo"], description=spec["description"], license="cc-by-4.0", tags=spec["tags"],
        private=False, exist_ok=True, min_fiftyone_version="1.21.0", preview_path=spec["preview"],
    )
    print("done: https://huggingface.co/datasets/dgural/" + spec["repo"])


if __name__ == "__main__":
    main(sys.argv[1])
