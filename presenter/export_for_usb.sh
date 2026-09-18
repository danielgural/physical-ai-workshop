#!/usr/bin/env bash
# Export both attendee datasets as FiftyOneDataset folders for the USB sticks.
#   ./presenter/export_for_usb.sh /Volumes/WORKSHOP
set -euo pipefail
DEST=${1:?dest dir}
python - "$DEST" <<'PY'
import sys, fiftyone as fo
dest = sys.argv[1]
for local, name in (("droid-mcap-workshop", "droid-mcap-workshop"), ("droid-frames", "droid-frames-workshop")):
    ds = fo.load_dataset(local)
    ds.export(export_dir=f"{dest}/{name}", dataset_type=fo.types.FiftyOneDataset, export_media=True)
    print(name, len(ds))
PY
pip download -r requirements.txt -d "$DEST/wheels" --quiet || true
echo "done -> $DEST"
