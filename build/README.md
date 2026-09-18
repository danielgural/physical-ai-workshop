# How the datasets were built (presenter-side)

Run from the repo root in a venv with `fiftyone>=1.21`, `mcap`, `mcap-protobuf-support`, `ultralytics`, `transformers`, `umap-learn`, `segno`, and `ffmpeg` on PATH. Source is the local `droid` dataset (100 episodes, `dgural/droid-mcap-demo`).

| Step | Script | Produces |
|---|---|---|
| 1 | `extract_frames.py --fps 1 --width 640` | `droid-frames`: 5,172 JPEGs + per-frame robot state (75 episodes have RGB video; 25 do not) |
| 2 | `select_episodes.py` | `episodes.json`: the 10-episode subset (RGB only, 6–40 s, planted episodes kept) |
| 3 | `build_mcap_subset.py` | `droid-mcap-workshop`: multimodal dataset, grasp/release temporal tags re-keyed |
| 4 | `bake_embeddings.py` | CLIP `clip`, `uniqueness`, `frames_sim`, `frames_viz`, `near-duplicate` tags |
| 5 | `autolabel.py` | `auto_labels` from Grounding DINO tiny (7 classes) |
| 6 | `export_yolo.py` | `yolo_export/` YOLOv5 format, split by episode, near-duplicates excluded from train |
| 7 | `../presenter/dryrun.py` | Nebius Serverless AI Job → `presenter/weights/best-<run>.pt` |
| 8 | `finalize.py --weights …` | `yolo11n_preds` on all frames, `eval_yolo` on val, `brick-visible` / `gripper-visible` temporal tags on the MCAP subset, `presenter/NUMBERS.md` |
| 9 | `push_to_hf.py mcap` / `frames` | `dgural/droid-mcap-workshop`, `dgural/droid-frames-workshop` |

`make_notebooks.py` regenerates `notebooks/0*.ipynb`; `../slides/build_deck.py` regenerates the decks; `../slides/make_qr.py` the QR codes.
