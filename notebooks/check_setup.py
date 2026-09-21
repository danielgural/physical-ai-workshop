"""One command to confirm the laptop is ready. Run after `download_data.py`.

    python notebooks/check_setup.py
"""

import shutil
import sys


def main():
    ok = True
    v = sys.version_info
    print(f"python {v.major}.{v.minor}.{v.micro}", "ok" if (3, 10) <= (v.major, v.minor) <= (3, 14) else "UNSUPPORTED (need 3.10–3.14)")
    ok &= (3, 10) <= (v.major, v.minor) <= (3, 14)

    try:
        import fiftyone as fo
        print(f"fiftyone {fo.__version__}", "ok" if fo.__version__ >= "1.21.0" else "TOO OLD (need >= 1.21.0)")
        ok &= fo.__version__ >= "1.21.0"
    except Exception as e:
        print("fiftyone import FAILED:", e); return 1

    for mod in ("ultralytics", "huggingface_hub", "google.protobuf"):
        try:
            __import__(mod); print(f"{mod} ok")
        except Exception as e:
            print(f"{mod} MISSING: {e}"); ok = False

    print("ffmpeg", "ok (optional)" if shutil.which("ffmpeg") else "not on PATH (only needed to re-extract frames; fine)")

    for name, n, mt in (("droid-mcap-workshop", 10, "multimodal"), ("droid-frames-workshop", 5172, "image")):
        if not fo.dataset_exists(name):
            print(f"{name} MISSING -> python notebooks/download_data.py"); ok = False; continue
        ds = fo.load_dataset(name)
        good = len(ds) == n and ds.media_type == mt
        print(f"{name}: {len(ds)} samples, {ds.media_type}", "ok" if good else f"UNEXPECTED (want {n}, {mt})")
        ok &= good
        import os
        if not os.path.exists(ds.first().filepath):
            print(f"  media missing on disk: {ds.first().filepath}"); ok = False

    print("\nREADY — run: jupyter lab notebooks/" if ok else "\nNOT READY — fix the lines above (or see SETUP.md)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
