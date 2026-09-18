"""Generate the QR codes the deck embeds. Dark-slide friendly: light modules on
transparent, plus a white-on-white version for printing.

    python slides/make_qr.py
"""

from pathlib import Path

import segno

OUT = Path(__file__).with_name("qr")
LINKS = {
    "repo": "https://github.com/danielgural/physical-ai-workshop",
    "selfserve": "https://app.voxel51.com",
    "nebius": "https://link.voxel51.com/nebius",
    "nebius-stuttgart": "https://link.voxel51.com/nebius-stuttgart",
    "nebius-munich": "https://link.voxel51.com/nebius-munich",
    "nebius-berlin": "https://link.voxel51.com/nebius-berlin",
    "hf-frames": "https://huggingface.co/datasets/dgural/droid-frames-workshop",
    "hf-mcap": "https://huggingface.co/datasets/dgural/droid-mcap-workshop",
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, url in LINKS.items():
        qr = segno.make(url, error="m")
        qr.save(OUT / f"{name}.png", scale=12, border=2, dark="#FFFFFF", light="#0B0F14")
        qr.save(OUT / f"{name}-print.png", scale=12, border=2, dark="#000000", light="#FFFFFF")
        print(f"{name:18s} {url}")


if __name__ == "__main__":
    main()
