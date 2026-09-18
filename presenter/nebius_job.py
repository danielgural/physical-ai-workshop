"""Submit, poll and collect a YOLO fine-tune on Nebius Serverless AI Jobs — no
plugin, no panel, just the Nebius Python SDK and an S3 client.

The training container is the public image that already speaks the
INPUT_S3_URI / OUTPUT_S3_URI / MODEL / TASK / HYPERPARAMS_JSON contract:

    ghcr.io/danielgural/fiftyone-yolo-train:latest

Data is staged on Nebius Object Storage (S3-compatible). API auth is an IAM
token minted by the `nebius` CLI; storage auth is a service-account access key
kept in a JSON file outside the repo.

    from presenter.nebius_job import NebiusJob
    job = NebiusJob()                       # reads env / key file
    job.upload_dir("build/yolo_export", "runs/stuttgart/input")
    job_id = job.submit("stuttgart", "runs/stuttgart", epochs=15)
    job.wait(job_id)                        # prints state transitions
    job.download("runs/stuttgart/output/best.pt", "presenter/weights/best.pt")
"""

import asyncio
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import boto3

PROJECT_ID = os.environ.get("NEBIUS_PROJECT_ID", "project-e00qxh26pr00nmpx9w83c1")
REGION = os.environ.get("NEBIUS_REGION", "eu-north1")
BUCKET = os.environ.get("NEBIUS_BUCKET", "physical-ai-workshop")
S3_ENDPOINT = f"https://storage.{REGION}.nebius.cloud"
IMAGE = os.environ.get("TRAIN_IMAGE", "ghcr.io/danielgural/fiftyone-yolo-train:latest")
KEY_FILE = os.environ.get("NEBIUS_S3_KEY_FILE", os.path.expanduser("~/Documents/api/nebius/workshop-s3-key.json"))

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}


def _iam_token():
    tok = os.environ.get("NEBIUS_IAM_TOKEN")
    if not tok:
        tok = subprocess.check_output(["nebius", "iam", "get-access-token"], text=True).strip()
        os.environ["NEBIUS_IAM_TOKEN"] = tok
    return tok


def _s3_creds():
    d = json.load(open(KEY_FILE))["status"]
    return d["aws_access_key_id"], d["secret"]


async def _first_subnet(sdk, project_id):
    """spec.subnet_id is required by the API; a project usually has exactly one."""
    from nebius.api.nebius.vpc.v1 import ListSubnetsRequest, SubnetServiceClient

    resp = await SubnetServiceClient(sdk).list(ListSubnetsRequest(parent_id=project_id))
    items = list(resp.items)
    if not items:
        raise RuntimeError(f"no VPC subnet in {project_id}; set NEBIUS_SUBNET_ID")
    return items[0].metadata.id


class NebiusJob:
    def __init__(self, project_id=PROJECT_ID, bucket=BUCKET, image=IMAGE):
        self.project_id, self.bucket, self.image = project_id, bucket, image
        _iam_token()
        key, secret = _s3_creds()
        self._key, self._secret = key, secret
        self.s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=REGION,
                               aws_access_key_id=key, aws_secret_access_key=secret)

    # --- storage -----------------------------------------------------------
    def upload_dir(self, local_dir, prefix, workers=32):
        """Parallel upload; ~10k small files take minutes, not an hour."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local_dir = Path(local_dir)
        files = [p for p in local_dir.rglob("*") if p.is_file()]

        def _put(p):
            self.s3.upload_file(str(p), self.bucket, f"{prefix}/{p.relative_to(local_dir).as_posix()}")

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_put, p) for p in files]):
                fut.result()
                done += 1
                if done % 1000 == 0 or done == len(files):
                    print(f"  uploaded {done}/{len(files)}", flush=True)
        return f"s3://{self.bucket}/{prefix}"

    def download(self, key, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(local_path))
        return local_path

    def list(self, prefix):
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]

    # --- jobs ----------------------------------------------------------------
    def submit(self, name, prefix, *, model="yolo11n.pt", epochs=15, imgsz=640, batch=32,
               platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb", timeout_h=2, disk_gb=100,
               extra_hyper=None):
        hyper = {"epochs": epochs, "imgsz": imgsz, "batch": batch, "patience": epochs}
        hyper.update(extra_hyper or {})
        env = {
            "INPUT_S3_URI": f"s3://{self.bucket}/{prefix}/input",
            "OUTPUT_S3_URI": f"s3://{self.bucket}/{prefix}/output",
            "AWS_S3_ENDPOINT": S3_ENDPOINT,
            "AWS_ACCESS_KEY_ID": self._key,
            "AWS_SECRET_ACCESS_KEY": self._secret,
            "AWS_REGION": REGION,
            "MODEL": model,
            "TASK": "detection",
            "HYPERPARAMS_JSON": json.dumps(hyper),
        }

        async def _do():
            from nebius.sdk import SDK
            from nebius.api.nebius.ai.v1 import CreateJobRequest, JobServiceClient, JobSpec
            from nebius.api.nebius.common.v1 import ResourceMetadata
            from nebius.api.nebius.compute.v1 import DiskSpec

            async with SDK() as sdk:
                subnet = os.environ.get("NEBIUS_SUBNET_ID") or await _first_subnet(sdk, self.project_id)
                spec = JobSpec(
                    image=self.image, platform=platform, preset=preset, subnet_id=subnet,
                    environment_variables=[JobSpec.EnvironmentVariable(name=k, value=v) for k, v in env.items()],
                    timeout=datetime.timedelta(hours=timeout_h),
                    disk=JobSpec.DiskSpec(size_bytes=disk_gb * 1024 ** 3, type=DiskSpec.DiskType.NETWORK_SSD),
                )
                req = CreateJobRequest(
                    metadata=ResourceMetadata(name=f"fiftyone-workshop-{name}"[:63], parent_id=self.project_id),
                    spec=spec)
                op = await JobServiceClient(sdk).create(req)
                return op.resource_id

        job_id = asyncio.run(_do())
        print(f"submitted {job_id}  ({platform}/{preset}, {model}, {epochs} epochs)")
        return job_id

    def state(self, job_id):
        async def _do():
            from nebius.sdk import SDK
            from nebius.api.nebius.ai.v1 import GetJobRequest, JobServiceClient

            async with SDK() as sdk:
                job = await JobServiceClient(sdk).get(GetJobRequest(id=job_id))
                st = job.status.state
                return getattr(st, "name", str(st).rsplit(".", 1)[-1])

        return asyncio.run(_do())

    def wait(self, job_id, every=15):
        t0, last = time.time(), None
        while True:
            st = self.state(job_id)
            if st != last:
                print(f"[{time.time() - t0:6.0f}s] {st}")
                last = st
            if st in TERMINAL:
                return st
            time.sleep(every)

    def cancel(self, job_id):
        async def _do():
            from nebius.sdk import SDK
            from nebius.api.nebius.ai.v1 import CancelJobRequest, JobServiceClient

            async with SDK() as sdk:
                await JobServiceClient(sdk).cancel(CancelJobRequest(id=job_id))

        asyncio.run(_do())

    def logs(self, job_id):
        """Nebius exposes job logs through the CLI; stream them here."""
        return subprocess.run(["nebius", "ai", "job", "logs", job_id, "--tail", "400"], text=True,
                              capture_output=True).stdout
