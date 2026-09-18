"""Non-interactive version of train_on_nebius.ipynb: export → stage → submit →
wait → download. Used for the pre-workshop dry run and as the fallback weights
producer.

    python presenter/dryrun.py --run dryrun --epochs 15
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presenter.nebius_job import NebiusJob  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dryrun")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    export = ROOT / "build" / "yolo_export"
    if not args.skip_export:
        subprocess.check_call([sys.executable, str(ROOT / "build" / "export_yolo.py"), "--out", str(export)])

    job = NebiusJob()
    t0 = time.time()
    if not args.skip_upload:
        print(job.upload_dir(export, f"runs/{args.run}/input"))
        print(f"upload {time.time() - t0:.0f}s")

    job_id = job.submit(args.run, f"runs/{args.run}", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, timeout_h=1)
    t1 = time.time()
    state = job.wait(job_id, every=15)
    print(f"job {state} after {time.time() - t1:.0f}s (submit→terminal)")
    if state != "COMPLETED":
        print(job.logs(job_id)[-4000:])
        sys.exit(1)
    out = job.download(f"runs/{args.run}/output/best.pt", ROOT / "presenter" / "weights" / f"best-{args.run}.pt")
    print("weights:", out, "| total wall", f"{time.time() - t0:.0f}s")
    print(job.logs(job_id)[-2500:])


if __name__ == "__main__":
    main()
