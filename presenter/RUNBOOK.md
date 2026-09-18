# Presenter runbook — Physical AI Workshop (120 min)

Rule: **nothing the room depends on is computed live.** Every brain run, label, prediction and evaluation ships inside the two HF datasets. The only live compute is the Nebius YOLO11n job, and `weights/best-dryrun.pt` is the fallback if it is late.

## Before the day

- [ ] Laptop: this repo's venv, `nebius profile create` done (`nebius iam whoami` works), `~/Documents/api/nebius/workshop-s3-key.json` present, Teams SDK venv for demo.fiftyone.ai (`~/Documents/development/gm/.venv`, key at `~/Documents/api/demo/demo.txt`).
- [ ] Local OSS datasets present: `droid-mcap-workshop` (10), `droid-frames` (5,172). `fiftyone app launch` once to warm the browser cache.
- [ ] demo.fiftyone.ai: `Droid demo` (100 episodes) and `Droid frames` (5,172 frames, all runs) load; Embeddings panel opens `frames_viz`; Model Evaluation opens `eval_yolo`.
- [ ] Nebius: `nebius ai job list` answers; bucket `physical-ai-workshop` lists `runs/dryrun/output/best.pt`; GPU quota for `gpu-l40s-a` in eu-north1.
- [ ] Deck for the city: `python slides/build_deck.py --city <city>`; all three QR codes scanned from a phone.
- [ ] USB sticks: `export_for_usb.sh` output (both datasets as FiftyOneDataset exports) + `requirements.txt` wheels.
- [ ] Screen recordings of segments 1, 2, 5 in `presenter/recordings/` (fallback if wifi or App misbehaves).
- [ ] One raw `.mcap` copied to the Desktop for the drag-and-drop hook.

## Timeline

| Min | Slide | Do | Fallback |
|---|---|---|---|
| 0–10 | 1–6 | Title. Repo QR up while people install. Framing. Loop. **Hook:** drag the Desktop `.mcap` into MCAP Explorer on demo.fiftyone.ai before slide 5. | recording-hook.mp4 |
| 10–25 | 7–8 | demo.fiftyone.ai → `Droid demo`. Grid previews, stream selector. Open `AUTOLab+0d4edc83+2023-10-21-19h-40m-44s` (hero: brick in drawer). Arrange wrist / ext1 / 3D, scrub, click a cuboid. Sidebar: task_type → success → sort duration_s asc → open the 1.4 s (`...20h-27m-01s`, gripper never closed) and 2.8 s (`...20h-43m-06s`, "success") episodes. Expand timeline tracks: grasp/release. Shift+T → drag → `arm-stall`. Notebook cell: `match_temporal_tags(tags="arm-stall")`. | recording-explore.mp4 |
| 25–40 | 9–10 | Show `build/extract_frames.py` for 60 s (H.264 → ffmpeg → JPEG, fields per frame). Switch to `Droid frames`. Filter `gripper_open == False`, `camera == wrist`. Sort by `uniqueness` asc (empty bench) then desc (hands, dropped objects). `near-duplicate` tag: 356 frames. Build and save `curated_train`. | recording-curate.mp4 |
| 40–55 | 11–12 | Embeddings panel → `frames_viz`. Color by camera (three worlds), task_type, gripper_open. Lasso an island. Similarity from the most-unique frame. Text prompt "a robot gripper holding a blue brick". Slide 12: batch on GPU, interactive on laptop. **Submit the Nebius job now** from `train_on_nebius.ipynb` (export + upload + submit cells; ~2 min). | — |
| 55–75 | 13–14 | `auto_labels`: count values; sort gripper boxes by low confidence (review queue); wrist-frame gripper coverage %. Split-by-episode. Slide 14. Check the job state in the Nebius console between points. | — |
| 75–95 | 15–17 | Job should be RUNNING/COMPLETED. Walk `train_on_nebius.ipynb`: spec, `wait`, `logs`, `download`, `apply_model` on val, `evaluate_detections`. Model Evaluation panel → `eval_yolo`, click confusion cells, sort val by `eval_yolo_fp`, missed grippers. | `weights/best-dryrun.pt` + the baked `eval_yolo`; say "this is Wednesday's run" |
| 95–105 | 18–19 | `06_close_the_loop.ipynb`: `intervals_from_predictions` → `brick-visible` tags → open an episode: grasp / release / brick-visible rows. Rank episodes by mean gripper confidence. Slide 19 payoff table. | baked `brick-visible` tags already on the dataset |
| 105–120 | 20–26 | Self-serve QR (slide 20). Hand to Nebius (21). Builder Program QR (25) — ask the room to scan *now*. Q&A on 26. | present 22–25 yourself |

## Planted values (verified on the local datasets 2026-09-18)

- 75 of 100 episodes carry RGB video; the other 25 have depth + projected point-cloud images only.
- Frames: 5,172 · cameras 1,724 each · gripper closed on 2,386 frames · 356 near-duplicates · uniqueness range 0.00–1.00.
- MCAP subset: 10 episodes, 891 MB, temporal tags grasp 17 / release 14.
- Hero episode `…19h-40m-44s`: 7.1 s, "put brick in drawer shelf and close drawer", success.
- Auto-labels / YOLO numbers: see `presenter/NUMBERS.md` (written by `build/finalize.py` after the dry run).

## If it goes wrong

- **App tile black for >10 s** — reload the page; the browser decodes H.264 in-page and the first episode primes it.
- **demo.fiftyone.ai slow** — switch to the local OSS App on the same datasets; the room is on the same thing anyway.
- **Nebius PROVISIONING for >5 min** — say it, keep going, come back at 85 min; if still not RUNNING, cancel and use the dry-run weights.
- **Wifi dead** — everything except the Nebius submit works offline: local datasets, local App, recordings.
- **Someone's `load_from_hub` fails** — USB import (`download_data.py --from-usb`).
