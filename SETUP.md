# Setup

Everything attendees run works on a laptop CPU. Do the download **before** you arrive.

## 1. Python environment

Python 3.10 or newer.

```bash
git clone https://github.com/danielgural/physical-ai-workshop
cd physical-ai-workshop
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pulls FiftyOne ≥ 1.21.0 (the multimodal media type), `ultralytics` for the optional training cell, and JupyterLab.

## 2. Download the datasets (~1.1 GB, once)

```bash
python notebooks/download_data.py
```

This loads two datasets from Hugging Face into your local FiftyOne database and keeps the media under `~/fiftyone/huggingface/`:

- `droid-mcap-workshop` — 10 robot episodes as `.mcap` (~0.9 GB)
- `droid-frames-workshop` — 5,172 frames with embeddings, labels and predictions (~0.2 GB)

If you are on hotel wifi, that is enough. If the venue download is slow, the presenter has both datasets on a USB stick; `download_data.py --from-usb /Volumes/WORKSHOP` imports them from there.

## 3. Check it works

```bash
python -c "import fiftyone as fo; print(fo.__version__); print(fo.list_datasets())"
```

You should see `1.21.x` or newer and both dataset names. Then:

```bash
jupyter lab notebooks/
```

Open `01_explore_mcap.ipynb` and run the first two cells. A browser tab with the FiftyOne App opens; click any episode.

## Troubleshooting

- **`protobuf` import error when opening an episode** — `pip install protobuf`; some Python builds miss it.
- **The App opens but episodes show a black tile** — give the first episode 5–10 s; the browser reads the MCAP file directly and decodes the video in-page.
- **`umap-learn` fails to build on Windows** — it is only needed if you recompute the visualization yourself. The precomputed UMAP in the dataset works without it.
- **Port 5151 already in use** — `fo.launch_app(dataset, port=5152)`.
- **Corporate laptop blocks Hugging Face** — use the USB import above.
- **Apple Silicon** — everything runs natively; the optional inference cells use the `mps` device automatically.

## Presenter-only pieces

`presenter/` needs a Nebius account (IAM token via the `nebius` CLI, an Object Storage access key) and the FiftyOne Enterprise SDK for `demo.fiftyone.ai`. Attendees never need any of it.
