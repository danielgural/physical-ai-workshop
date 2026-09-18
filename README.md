# Physical AI Workshop — from raw robot logs to a trained detector

Hands-on companion for the **Nebius × Voxel51 Physical AI Workshop** (Stuttgart · Munich · Berlin, September 2026, and the US roadshow that follows).

In two hours we take real robot recordings — [DROID](https://droid-dataset.github.io) teleop episodes stored as MCAP — and run the whole loop in open-source [FiftyOne](https://github.com/voxel51/fiftyone):

**see it → curate it → embed it → auto-label it → train on it → evaluate → back to the timeline**

The presenter runs the GPU steps on **Nebius Serverless AI Jobs**. You follow along on your laptop, on CPU, with the results of every GPU step already baked into the datasets you download. Nothing here needs a GPU or a cloud account.

## Setup (do this before the workshop — the venue wifi will not enjoy 1 GB × 60 people)

```bash
git clone https://github.com/danielgural/physical-ai-workshop
cd physical-ai-workshop
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
python notebooks/download_data.py                       # ~1.1 GB from Hugging Face, once
jupyter lab notebooks/
```

Full details, Windows notes and troubleshooting: [SETUP.md](SETUP.md).

## Agenda → notebook

| Min | Segment | Notebook | What you do |
|---|---|---|---|
| 0–10 | Framing | — | slides |
| 10–25 | **Explore** a DROID recording in FiftyOne's multimodal MCAP viewer | [`01_explore_mcap.ipynb`](notebooks/01_explore_mcap.ipynb) | 10 episodes, 3 cameras + 3D on one clock; filter, tag intervals |
| 25–40 | **Curate**: frames out of the recording, filter, dedupe | [`02_curate_frames.ipynb`](notebooks/02_curate_frames.ipynb) | 5k frames from 75 episodes; gripper state, uniqueness, near-duplicates |
| 40–55 | **Embeddings**, similarity, visualization | [`03_embeddings.ipynb`](notebooks/03_embeddings.ipynb) | UMAP + similarity index are precomputed; lasso and sort |
| 55–75 | **Auto-label** with open-vocabulary detection | [`04_autolabel.ipynb`](notebooks/04_autolabel.ipynb) | inspect Grounding DINO labels; optional: run it on 20 frames |
| 75–95 | **Train** YOLO11n (presenter: Nebius L40S) | [`05_train_eval.ipynb`](notebooks/05_train_eval.ipynb) | predictions + evaluation are baked in; optional: 1-epoch CPU train |
| 95–105 | **Close the loop** on the MCAP timeline | [`06_close_the_loop.ipynb`](notebooks/06_close_the_loop.ipynb) | detector output as temporal tags on the episodes |
| 105–120 | Nebius: Serverless, Token Factory, Builder Program | — | slides |

## The data

Two FiftyOne datasets on Hugging Face, both derived from DROID (CC-BY-4.0):

| Dataset | What | Size |
|---|---|---|
| [`dgural/droid-mcap-workshop`](https://huggingface.co/datasets/dgural/droid-mcap-workshop) | 10 episodes as multimodal `.mcap` samples — 3 cameras, point clouds, cuboids, joint state, grasp/release temporal tags, episode fields | ~0.9 GB |
| [`dgural/droid-frames-workshop`](https://huggingface.co/datasets/dgural/droid-frames-workshop) | 5,172 RGB frames (1 fps, 640 px) from the 75 episodes with video, with gripper state per frame, CLIP embeddings, uniqueness, near-duplicate tags, UMAP + similarity runs, Grounding DINO auto-labels, YOLO11n predictions and an evaluation run | ~0.2 GB |

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

episodes = fouh.load_from_hub("dgural/droid-mcap-workshop")
frames = fouh.load_from_hub("dgural/droid-frames-workshop")
fo.launch_app(episodes)
```

The full 100-episode set is [`dgural/droid-mcap-demo`](https://huggingface.co/datasets/dgural/droid-mcap-demo) (~11 GB).

## What's where

```
notebooks/    attendee notebooks 01–06 + download_data.py
build/        how the datasets were made (frame extraction from MCAP, embeddings, auto-labels, subset)
presenter/    the Nebius training notebook, job helper and runbook
slides/       deck generator + QR codes
```

`build/extract_frames.py` is the one piece attendees ask for most: decoding the H.264 camera streams inside an MCAP into a FiftyOne image dataset that remembers episode, camera and timestamp for every frame.

## Requirements

FiftyOne ≥ 1.21.0 — the first open-source release with the `multimodal` media type, where a sample whose filepath is an `.mcap` *is* the episode. CPU is enough for everything attendees run.

## Links

- FiftyOne: [github.com/voxel51/fiftyone](https://github.com/voxel51/fiftyone) · [docs.voxel51.com](https://docs.voxel51.com)
- Try the hosted version: [app.voxel51.com](https://app.voxel51.com)
- Nebius Builder Program: [link.voxel51.com/nebius](https://link.voxel51.com/nebius)
