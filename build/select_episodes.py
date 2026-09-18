"""Pick the workshop's MCAP episode subset.

Ten episodes from the local 100-episode `droid` dataset, restricted to the
ones that carry RGB camera video (25 of the 100 only have depth/point-cloud
streams). The subset mixes task families and outcomes and keeps the planted
"look at this one" episodes the presenter runbook relies on.

    python build/select_episodes.py            # writes build/episodes.json
"""

import json
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

PLANTED = [
    "AUTOLab+0d4edc83+2023-10-21-20h-27m-01s",  # 1.4 s, gripper never closed, failed
    "AUTOLab+0d4edc83+2023-10-27-20h-43m-06s",  # 2.8 s degenerate "success"
    "AUTOLab+0d4edc83+2023-10-21-19h-24m-27s",  # highest peak joint speed among RGB episodes
    "AUTOLab+0d4edc83+2023-10-21-19h-40m-44s",  # clean brick-in-drawer success (hero episode)
]
N = 10


def main():
    droid = fo.load_dataset("droid")
    frames = fo.load_dataset("droid-frames")
    rgb = set(frames.distinct("episode_id"))
    view = droid.match(F("episode_id").is_in(list(rgb)))

    chosen = [e for e in PLANTED if e in rgb]
    # Fill by task family, alternating success/failure. Prefer episodes between
    # 6 and 40 s: long enough to scrub, short enough that the whole subset stays
    # around 1 GB for attendee downloads (the 70 s clump-unclump episodes are
    # ~300 MB each).
    for task_type in ["fold-cloth", "unhang-cloth", "clump-unclump", "container-transfer", "multi-step"]:
        for success in (False, True):
            if len(chosen) >= N:
                break
            cands = (view.match((F("task_type") == task_type) & (F("success") == success))
                     .match(~F("episode_id").is_in(chosen))
                     .match((F("duration_s") >= 6) & (F("duration_s") <= 40))
                     .sort_by("duration_s"))
            s = cands.first() if len(cands) else None
            if s is not None:
                chosen.append(s.episode_id)
    chosen = chosen[:N]

    sub = droid.match(F("episode_id").is_in(chosen))
    rows = [{"episode_id": s.episode_id, "filepath": s.filepath, "task_type": s.task_type,
             "success": s.success, "duration_s": s.duration_s, "task": s.current_task}
            for s in sub.select_fields(["episode_id", "task_type", "success", "duration_s", "current_task"])]
    out = Path(__file__).with_name("episodes.json")
    out.write_text(json.dumps(rows, indent=2))
    total_mb = sum(Path(r["filepath"]).stat().st_size for r in rows) / 1e6
    for r in rows:
        print(f"{r['episode_id']}  {r['task_type']:18s} success={r['success']!s:5s} {r['duration_s']:6.1f}s")
    print(f"{len(rows)} episodes, {total_mb:.0f} MB of MCAP -> {out}")


if __name__ == "__main__":
    main()
