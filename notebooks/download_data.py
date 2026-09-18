"""Download the two workshop datasets from Hugging Face into local FiftyOne.

    python notebooks/download_data.py               # from Hugging Face
    python notebooks/download_data.py --from-usb /Volumes/WORKSHOP
"""

import argparse
import sys

import fiftyone as fo

DATASETS = {
    "droid-mcap-workshop": "dgural/droid-mcap-workshop",
    "droid-frames-workshop": "dgural/droid-frames-workshop",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-usb", metavar="DIR", help="import FiftyOneDataset exports from a folder instead")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    for name, repo in DATASETS.items():
        if fo.dataset_exists(name) and not args.force:
            print(f"{name}: already here ({len(fo.load_dataset(name))} samples)")
            continue
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
        if args.from_usb:
            ds = fo.Dataset.from_dir(dataset_dir=f"{args.from_usb}/{name}",
                                     dataset_type=fo.types.FiftyOneDataset, name=name, persistent=True)
        else:
            import fiftyone.utils.huggingface as fouh
            ds = fouh.load_from_hub(repo, name=name, persistent=True)
        print(f"{name}: {len(ds)} samples, media type {ds.media_type}")

    print("\nReady. Next:  jupyter lab notebooks/")


if __name__ == "__main__":
    sys.exit(main())
