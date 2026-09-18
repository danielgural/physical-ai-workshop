"""Extract RGB frames from DROID MCAP episodes into an image dataset.

Each MCAP carries three H.264 camera streams (foxglove.CompressedVideo on
/camera/{wrist,ext1,ext2}/image_rgb), robot state (robot.GripperState,
robot.Scalar) and episode metadata. This decodes every camera stream through
ffmpeg at a fixed sample rate, writes JPEGs, and builds a FiftyOne image
dataset whose samples remember which episode, camera and timestamp they came
from — so predictions on frames can be mapped back onto the MCAP timeline.

    python build/extract_frames.py --source droid --name droid-frames --fps 1.0 --width 640

Presenter-side. Attendees load the result from Hugging Face.
"""

import argparse
import bisect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fiftyone as fo
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

VIDEO_SCHEMA = "foxglove.CompressedVideo"
GRIPPER_TOPIC = "/robot/gripper_state"
SPEED_TOPIC = "/robot/joint_speed_norm"
EE_SPEED_TOPIC = "/robot/ee/linear_speed"

EPISODE_FIELDS = [
    "episode_id", "lab", "current_task", "task_type", "success",
    "duration_s", "peak_joint_speed", "gripper_open_frac", "user",
]


def _nearest(times, values, t):
    """Value whose timestamp is nearest to t (times sorted)."""
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    if i == 0:
        return values[0]
    if i == len(times):
        return values[-1]
    return values[i] if times[i] - t < t - times[i - 1] else values[i - 1]


def decode_episode(mcap_path, out_dir, fps, width):
    """Decode every RGB camera stream; return per-frame records."""
    streams, gripper, speed, ee_speed = {}, [], [], []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        video_topics = sorted(
            ch.topic for ch in summary.channels.values()
            if summary.schemas[ch.schema_id].name == VIDEO_SCHEMA
        )
        for _, ch, msg, proto in reader.iter_decoded_messages(
            topics=[*video_topics, GRIPPER_TOPIC, SPEED_TOPIC, EE_SPEED_TOPIC]
        ):
            if ch.topic == GRIPPER_TOPIC:
                gripper.append((msg.log_time, bool(proto.open)))
            elif ch.topic == SPEED_TOPIC:
                speed.append((msg.log_time, float(proto.value)))
            elif ch.topic == EE_SPEED_TOPIC:
                ee_speed.append((msg.log_time, float(proto.value)))
            else:
                streams.setdefault(ch.topic, []).append((msg.log_time, bytes(proto.data)))

    g_t, g_v = zip(*gripper) if gripper else ([], [])
    s_t, s_v = zip(*speed) if speed else ([], [])
    e_t, e_v = zip(*ee_speed) if ee_speed else ([], [])

    records = []
    for topic, packets in streams.items():
        camera = topic.split("/")[2]  # /camera/<cam>/image_rgb
        times = [t for t, _ in packets]
        t0 = times[0]
        cam_dir = Path(out_dir) / camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp()
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "h264", "-i", "pipe:0",
             "-vf", f"fps={fps},scale={width}:-2", "-q:v", "3",
             os.path.join(tmp, "f_%05d.jpg")],
            stdin=subprocess.PIPE,
        )
        for _, data in packets:
            proc.stdin.write(data)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {mcap_path} {topic}")

        for i, name in enumerate(sorted(os.listdir(tmp))):
            # ffmpeg's fps filter emits frame i at t0 + i/fps on the stream clock
            t = t0 + int(i * 1e9 / fps)
            dst = cam_dir / f"{i:05d}.jpg"
            shutil.move(os.path.join(tmp, name), dst)
            records.append({
                "filepath": str(dst),
                "camera": camera,
                "frame_index": i,
                "timestamp_ns": int(t),
                "t_rel_s": round((t - t0) / 1e9, 3),
                "gripper_open": _nearest(g_t, g_v, t),
                "joint_speed_norm": _nearest(s_t, s_v, t),
                "ee_linear_speed": _nearest(e_t, e_v, t),
            })
        shutil.rmtree(tmp, ignore_errors=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="droid", help="multimodal dataset of MCAP episodes")
    ap.add_argument("--name", default="droid-frames")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "frames"))
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    source = fo.load_dataset(args.source)
    if fo.dataset_exists(args.name):
        fo.delete_dataset(args.name)
    dataset = fo.Dataset(args.name, persistent=True)

    episodes = source.select_fields(EPISODE_FIELDS)
    if args.limit:
        episodes = episodes.limit(args.limit)

    total = 0
    for ep in episodes:
        ep_dir = Path(args.out) / ep.episode_id
        records = decode_episode(ep.filepath, ep_dir, args.fps, args.width)
        samples = []
        for r in records:
            s = fo.Sample(filepath=r.pop("filepath"))
            for k, v in r.items():
                s[k] = v
            s["episode_id"] = ep.episode_id
            s["mcap_sample_id"] = ep.id
            s["mcap_filename"] = Path(ep.filepath).name
            for k in EPISODE_FIELDS[1:]:
                s[f"episode_{k}" if k in ("duration_s", "peak_joint_speed", "gripper_open_frac") else k] = ep[k]
            samples.append(s)
        dataset.add_samples(samples)
        total += len(samples)
        print(f"{ep.episode_id}: {len(samples)} frames (total {total})", flush=True)

    dataset.compute_metadata()
    dataset.info = {
        "source": "DROID (droid-dataset.github.io), CC-BY-4.0",
        "fps": args.fps,
        "width": args.width,
        "note": "frames decoded from foxglove.CompressedVideo camera streams in the MCAP episodes",
    }
    dataset.save()
    print(f"done: {dataset.name} — {len(dataset)} frames from {len(episodes)} episodes")


if __name__ == "__main__":
    main()
