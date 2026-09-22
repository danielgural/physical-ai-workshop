"""Build the workshop deck. Dark theme, orange accent, speaker notes on every slide.

    python slides/build_deck.py --city stuttgart      # -> slides/physical-ai-workshop-stuttgart.pptx
    python slides/build_deck.py --all
"""

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).parent
ASSETS, QR = HERE / "assets", HERE / "qr"

BG = RGBColor(0x0B, 0x0F, 0x14)
PANEL = RGBColor(0x14, 0x1A, 0x22)
INK = RGBColor(0xF2, 0xF4, 0xF7)
DIM = RGBColor(0x9A, 0xA4, 0xB2)
ACCENT = RGBColor(0xFF, 0x6D, 0x04)      # Voxel51 orange
ACCENT2 = RGBColor(0x5B, 0xC0, 0xEB)
NEBIUS = RGBColor(0x7C, 0xF2, 0x9A)      # Nebius-ish green for the partner segment
LINE = RGBColor(0x23, 0x2A, 0x38)

CITIES = {
    "stuttgart": dict(name="Stuttgart", date="22 September 2026", venue="Impact Hub Stuttgart", qr="nebius-stuttgart"),
    "munich": dict(name="Munich", date="23 September 2026", venue="Impact Hub Munich", qr="nebius-munich"),
    "berlin": dict(name="Berlin", date="26 September 2026", venue="MotionLab Berlin", qr="nebius-berlin"),
    "generic": dict(name="Physical AI Roadshow", date="2026", venue="", qr="nebius"),
}

LOOP = ["see", "curate", "embed", "label", "train", "evaluate"]


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # --- primitives ---------------------------------------------------------
    def slide(self, notes=""):
        s = self.prs.slides.add_slide(self.blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = BG
        self.n += 1
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        # footer
        self.text(s, Inches(0.6), Inches(7.0), Inches(8), Inches(0.35),
                  "Nebius × Voxel51 · Physical AI Workshop", size=11, color=DIM)
        self.text(s, Inches(12.0), Inches(7.0), Inches(0.8), Inches(0.35), str(self.n), size=11, color=DIM,
                  align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, x, y, w, h, content, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, font="Helvetica Neue",
             anchor=MSO_ANCHOR.TOP):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        lines = content if isinstance(content, list) else [content]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            runs = line if isinstance(line, list) else [(line, color, bold)]
            for t, c, b in runs:
                r = p.add_run()
                r.text = t
                r.font.size = Pt(size)
                r.font.bold = b
                r.font.color.rgb = c
                r.font.name = font
        return tb

    def kicker(self, s, label, color=ACCENT):
        self.text(s, Inches(0.9), Inches(0.55), Inches(8), Inches(0.4), label.upper(), size=13, color=color, bold=True)

    def title(self, s, t, size=40):
        self.text(s, Inches(0.9), Inches(0.95), Inches(11.5), Inches(1.1), t, size=size, color=INK, bold=True)

    def bullets(self, s, items, y=Inches(2.3), size=21, w=Inches(11.5), h=None):
        lines = []
        for it in items:
            if isinstance(it, tuple):
                lines.append([("•  ", ACCENT, True), (it[0], INK, True), (it[1], DIM, False)])
            else:
                lines.append([("•  ", ACCENT, True), (it, INK, False)])
        if h is None:
            h = min(Inches(4.2), Inches(7.0) - y)
        tb = self.text(s, Inches(0.9), y, w, h, lines, size=size)
        for p in tb.text_frame.paragraphs:
            p.space_after = Pt(12)
        return tb

    def panel(self, s, x, y, w, h, fill=PANEL):
        r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        r.fill.solid()
        r.fill.fore_color.rgb = fill
        r.line.color.rgb = LINE
        r.adjustments[0] = 0.04
        r.shadow.inherit = False
        return r

    def code(self, s, x, y, w, h, lines, size=16):
        box = self.panel(s, x, y, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.2)
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = line or " "
            r.font.size = Pt(size)
            r.font.name = "Menlo"
            r.font.color.rgb = RGBColor(0xC8, 0xD3, 0xE0) if not line.lstrip().startswith("#") else DIM

    def image(self, s, path, x, y, w=None, h=None):
        p = Path(path)
        if not p.exists():
            box = self.panel(s, x, y, w or Inches(5), h or Inches(3))
            box.text_frame.text = f"[missing: {p.name}]"
            return box
        kw = {}
        if w is not None:
            kw["width"] = w
        if h is not None:
            kw["height"] = h
        return s.shapes.add_picture(str(p), x, y, **kw)

    def qr(self, s, name, x, y, size=Inches(2.6), caption=None):
        self.image(s, QR / f"{name}.png", x, y, w=size, h=size)
        if caption:
            cw = size + Inches(2)
            cx = min(x - Inches(1), Inches(13.333) - cw - Inches(0.2))
            self.text(s, cx, y + size + Inches(0.05), cw, Inches(0.4), caption, size=13,
                      color=DIM, align=PP_ALIGN.CENTER, font="Menlo")

    def loop_row(self, s, y, hot=None, x0=Inches(0.9), gap=Inches(0.25)):
        n = len(LOOP)
        total_w = Inches(11.5)
        w = (total_w - gap * (n - 1)) / n
        for i, step in enumerate(LOOP):
            x = x0 + (w + gap) * i
            is_hot = hot is not None and (i == hot if isinstance(hot, int) else i in hot)
            box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, Inches(0.75))
            box.fill.solid()
            box.fill.fore_color.rgb = ACCENT if is_hot else PANEL
            box.line.color.rgb = ACCENT if is_hot else LINE
            box.adjustments[0] = 0.2
            tf = box.text_frame
            tf.text = step
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.runs[0].font.size = Pt(18)
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = BG if is_hot else INK
            if i < n - 1:
                self.text(s, x + w - Inches(0.05), y + Inches(0.15), gap + Inches(0.1), Inches(0.5), "›", size=22,
                          color=DIM, align=PP_ALIGN.CENTER)

    def segment(self, k, name, minutes, sub, notes, hot):
        """Section divider: agenda position + which loop step is lit."""
        s = self.slide(notes)
        self.loop_row(s, Inches(0.9), hot=hot)
        self.text(s, Inches(0.9), Inches(2.6), Inches(11.5), Inches(0.5), f"{minutes}", size=16, color=ACCENT, bold=True,
                  font="Menlo")
        self.text(s, Inches(0.9), Inches(3.0), Inches(11.5), Inches(1.3), name, size=54, color=INK, bold=True)
        self.text(s, Inches(0.9), Inches(4.4), Inches(11.5), Inches(0.9), sub, size=22, color=DIM)
        return s

    def follow_along(self, s, notebook, live):
        """Bottom strip: attendee notebook vs presenter backend."""
        y = Inches(6.15)
        self.panel(s, Inches(0.9), y, Inches(11.5), Inches(0.7))
        self.text(s, Inches(1.1), y + Inches(0.14), Inches(5.6), Inches(0.5),
                  [[("you  ", ACCENT, True), (notebook, INK, False)]], size=15, font="Menlo")
        self.text(s, Inches(6.8), y + Inches(0.14), Inches(5.5), Inches(0.5),
                  [[("presenter  ", NEBIUS, True), (live, INK, False)]], size=15, font="Menlo")


def build(city_key):
    city = CITIES[city_key]
    d = Deck()

    # 1 title
    s = d.slide("Start on time. Repo QR is on the next slide; people will still be installing. "
                "Say the Nebius Builder Program early: it comes back at the end with a city-specific QR.")
    d.kicker(s, f"Nebius × Voxel51  ·  {city['name']}  ·  {city['date']}")
    d.text(s, Inches(0.9), Inches(1.7), Inches(11.5), Inches(2.4),
           ["From raw robot logs", "to a trained detector"], size=60, color=INK, bold=True)
    d.text(s, Inches(0.9), Inches(4.3), Inches(10.5), Inches(1.0),
           "A hands-on Physical AI workshop: real DROID recordings, open-source FiftyOne, GPUs on Nebius.",
           size=22, color=DIM)
    d.text(s, Inches(0.9), Inches(6.2), Inches(10), Inches(0.5),
           f"Daniel Gural — Partnership Engineering, Voxel51{'  ·  ' + city['venue'] if city['venue'] else ''}",
           size=16, color=DIM)

    # 2 follow along
    s = d.slide("Leave this up during the whole setup chatter. The download is 1.1 GB; anyone who did not "
                "pre-download should start now and can watch the first two segments. USB sticks are at the front.")
    d.kicker(s, "Follow along")
    d.title(s, "Everything you need is one repo away")
    d.code(s, Inches(0.9), Inches(2.3), Inches(7.6), Inches(2.9), [
        "git clone https://github.com/danielgural/physical-ai-workshop",
        "cd physical-ai-workshop",
        "python -m venv .venv && source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python notebooks/download_data.py     # 1.1 GB, once",
        "jupyter lab notebooks/",
    ])
    d.bullets(s, [
        ("Laptop CPU is enough. ", "Every GPU step's output is already in the datasets you download."),
        ("Slow wifi? ", "USB sticks at the front: python notebooks/download_data.py --from-usb /Volumes/WORKSHOP"),
    ], y=Inches(5.35), size=17, w=Inches(7.6))
    d.qr(s, "repo", Inches(9.4), Inches(2.3), size=Inches(3.0), caption="github.com/danielgural/physical-ai-workshop")

    # 3 framing: the data problem
    s = d.slide("Land the question: when did you last watch 20 random episodes from your training set? "
                "Most robotics teams cannot answer. That is the whole workshop.")
    d.kicker(s, "Why we are here")
    d.title(s, "Physical AI has a data problem. It is not collection.")
    d.bullets(s, [
        ("Every demonstration is expensive ", "— human teleop time; you cannot just collect more."),
        ("Quality lives at the episode level ", "— failed grasps, occluded cameras, teleop glitches, degenerate 2-second 'successes'."),
        ("Nobody looks ", "— a recording is a 60 MB binary blob; without tooling it never gets opened."),
        ("The model is rarely the bottleneck ", "— whether anyone has looked at the data is."),
    ])

    # 4 the loop
    s = d.slide("Anchor diagram. Each segment lights one step. Say the split now: I run the GPU steps on Nebius; "
                "you run everything else on your laptop with the GPU outputs baked in.")
    d.kicker(s, "Today")
    d.title(s, "The loop every serious robotics team runs")
    d.loop_row(s, Inches(2.4))
    d.text(s, Inches(0.9), Inches(3.6), Inches(11.5), Inches(1.5),
           "100 DROID robot-arm episodes. Two hours. Open-source FiftyOne on your laptop for the interactive steps; "
           "Nebius Serverless AI Jobs for the GPU steps. The same loop, pointed at millions of episodes, is what "
           "production teams run.", size=21, color=DIM)
    d.panel(s, Inches(0.9), Inches(5.2), Inches(5.6), Inches(1.2))
    d.text(s, Inches(1.1), Inches(5.3), Inches(5.3), Inches(1.0),
           [[("you  ", ACCENT, True), ("explore · curate · browse embeddings · review labels · evaluate · tag", INK, False)]],
           size=15)
    d.panel(s, Inches(6.8), Inches(5.2), Inches(5.6), Inches(1.2))
    d.text(s, Inches(7.0), Inches(5.3), Inches(5.3), Inches(1.0),
           [[("Nebius  ", NEBIUS, True), ("embeddings at scale · YOLO11n fine-tune · (Token Factory: auto-label VLMs)", INK, False)]],
           size=15)

    # 5 what a recording is
    s = d.slide("DEMO: drag a raw .mcap into the MCAP Explorer panel BEFORE clicking to this slide. "
                "'Nothing was uploaded or converted — the browser is reading the file.'")
    d.kicker(s, "The data")
    d.title(s, "A robot demonstration is not an image")
    d.bullets(s, [
        ("3 cameras ", "— wrist + two externals, H.264, synchronized"),
        ("Depth + fused point clouds ", "— the scene in 3D"),
        ("Joint state, gripper state, end-effector speed ", "— what the arm actually did"),
        ("3D cuboid tracks ", "— labels over time"),
        ("One .mcap file ", "— one of 76,000 in DROID, 13 labs"),
    ], size=19, w=Inches(6.4))
    d.image(s, ASSETS / "episode-viewer-3d.png", Inches(7.5), Inches(2.3), w=Inches(5.2))

    # 6 the trick
    s = d.slide("The only code slide before the notebooks. FiftyOne ≥ 1.21: multimodal media type. "
                "The file is the sample.")
    d.kicker(s, "The whole trick")
    d.title(s, "An episode is a sample")
    d.code(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(2.3), [
        "import fiftyone as fo",
        "",
        'dataset = fo.Dataset("droid")',
        'dataset.add_samples([fo.Sample("episode-0001.mcap"), ...])',
        "",
        'dataset.media_type   # "multimodal" — no conversion, no ROS install',
    ])
    d.text(s, Inches(0.9), Inches(4.9), Inches(11.5), Inches(1.2),
           "Cameras, point clouds, transforms and cuboids are discovered from the file. Fields you add — task, lab, "
           "success, joint statistics — make episodes queryable. Temporal tags label a time range, not a frame.",
           size=20, color=DIM)

    # 7 segment: see
    s = d.segment("see", "Explore a recording", "10–25 min", "Three cameras and 3D on one clock. Filter episodes like rows. Tag an interval.",
                  "Grid previews → open the hero episode (19h-40m-44s) → arrange wrist / ext / 3D → scrub → click a cuboid → "
                  "sidebar filter task_type, success → sort duration_s asc → open the 1.4 s and 2.8 s episodes → "
                  "expand timeline: grasp/release tags → Shift+T, tag arm-stall → match_temporal_tags in the notebook.", hot=0)
    d.follow_along(s, "01_explore_mcap.ipynb", "demo.fiftyone.ai · Droid demo (100 episodes)")

    # 8 planted moments
    s = d.slide("What the room should find in the 10 episodes they have. Do not reveal until they have looked.")
    d.kicker(s, "See")
    d.title(s, "What is in the ten episodes you downloaded")
    d.bullets(s, [
        ("A 1.4 s 'multi-step task' ", "— gripper never closed (gripper_open_frac = 1.0). Marked failed. Correctly."),
        ("A 2.8 s 'success' ", "— put brick in drawer and close it, in under three seconds? Look at it."),
        ("grasp / release intervals ", "— 31 temporal tags already on the timeline; queryable with match_temporal_tags()."),
        ("Five task families ", "— brick-in-drawer, fold cloth, unhang cloth, clump/unclump, multi-step."),
    ], size=19, w=Inches(6.6))
    d.image(s, ASSETS / "droid-temporal-tags.png", Inches(7.7), Inches(2.3), w=Inches(5.0))

    # 9 segment: curate
    s = d.segment("curate", "Curate: frames out of the recording", "25–40 min",
                  "Decode the camera streams at 1 fps. Filter with robot state. Drop near-duplicates. Save the view.",
                  "Show extract_frames.py briefly: H.264 NAL units → ffmpeg → JPEG, each frame keeps episode_id, camera, "
                  "timestamp_ns, gripper_open, joint_speed_norm. Then in the App: match gripper_open == False, sort by "
                  "uniqueness both ways, show the near-duplicate tag, build + save curated_train.", hot=1)
    d.follow_along(s, "02_curate_frames.ipynb", "demo.fiftyone.ai · Droid workshop frames (5,172 frames)")

    # 10 frames detail
    s = d.slide("The frame remembers where it came from. That link is the payoff in segment 6.")
    d.kicker(s, "Curate")
    d.title(s, "Every frame remembers its recording")
    d.code(s, Inches(0.9), Inches(2.3), Inches(6.6), Inches(3.4), [
        "# build/extract_frames.py",
        "for _, ch, msg, proto in reader.iter_decoded_messages(",
        '        topics=["/camera/ext1/image_rgb"]):',
        "    ffmpeg.stdin.write(proto.data)   # H.264 → fps=1 → 640px JPEG",
        "",
        "sample = fo.Sample(filepath=jpg)",
        'sample["episode_id"]   = episode.episode_id',
        'sample["camera"]       = "ext1"',
        'sample["timestamp_ns"] = t',
        'sample["gripper_open"] = nearest(gripper_state, t)',
    ], size=14)
    d.image(s, ASSETS / "frame-ext1.jpg", Inches(7.8), Inches(2.3), w=Inches(4.7))
    d.image(s, ASSETS / "frame-wrist.jpg", Inches(7.8), Inches(4.5), w=Inches(2.3))
    d.text(s, Inches(10.2), Inches(4.5), Inches(2.4), Inches(1.6),
           "5,172 frames · 75 episodes · 3 cameras · gripper state and joint speed per frame", size=14, color=DIM)

    # 11 segment: embed
    s = d.segment("embed", "Embeddings, similarity, visualization", "40–55 min",
                  "CLIP on every frame. UMAP to see structure. A similarity index for 'more like this' — and text prompts.",
                  "Embeddings panel: color by camera (3 worlds), task_type, gripper_open. Lasso an island. Similarity from "
                  "the most-unique frame. Text prompt 'a robot gripper holding a blue brick'. Say clearly: on the deployment "
                  "this compute is a Nebius Serverless job; the interactive part is what humans do.", hot=2)
    d.follow_along(s, "03_embeddings.ipynb  (runs precomputed)", "Nebius Serverless AI Jobs · embeddings at scale")

    # 12 embeddings as a job
    s = d.slide("One slide on the split: batch compute lives in a GPU job, humans get the interactive part. "
                "If the stretch embeddings job landed, show its console entry here.")
    d.kicker(s, "Embed", color=NEBIUS)
    d.title(s, "Batch on the GPU, interactive on the laptop")
    d.bullets(s, [
        ("compute_embeddings ", "— one call, any model: CLIP today, Qwen3-VL-Embedding or SigLIP2 tomorrow"),
        ("As a Nebius Serverless AI Job ", "— a container reads frames from object storage, writes vectors back; no cluster to keep warm"),
        ("Then the brain runs ", "— compute_similarity, compute_visualization, compute_uniqueness, find_duplicates"),
        ("The room never waits on a GPU ", "— every run ships in the dataset; you lasso, sort, tag"),
    ], size=19)
    d.code(s, Inches(0.9), Inches(5.2), Inches(11.5), Inches(0.9), [
        'frames.compute_embeddings(model, embeddings_field="clip");  fob.compute_visualization(frames, embeddings="clip", brain_key="frames_viz")'
    ], size=14)

    # 13 segment: label
    s = d.segment("label", "Auto-label with open vocabulary", "55–75 min",
                  "Grounding DINO, prompted with words. 7 labels on 5,172 frames. Humans review the queue, not the corpus.",
                  "Show the prompt string. Count values. Sort gripper boxes by low confidence → this is the review queue. "
                  "Sanity check: % of wrist frames with a gripper box. Then the split-by-episode rule. "
                  "Mention Token Factory VLMs as the natural next labeler (Nebius closes on that).", hot=3)
    d.follow_along(s, "04_autolabel.ipynb", "Grounding DINO on the presenter GPU · Token Factory VLMs next")

    # 14 label detail
    s = d.slide("The 'models draft, humans verify' slide.")
    d.kicker(s, "Label")
    d.title(s, "Models draft. Humans verify the 5% that matters.")
    d.bullets(s, [
        ("Prompt, do not annotate ", "— 'robot gripper . robot arm . blue brick . scissors . drawer . cloth . cardboard box'"),
        ("Confidence is a field ", "— sort by it; the bottom of the list is your review queue"),
        ("Check labels against state ", "— the gripper is in every wrist frame; if boxes are missing, the labeler is wrong, not the data"),
        ("Split by episode, never by frame ", "— 1 fps neighbours are near-identical; a frame split leaks"),
        ("Curation chose what to label ", "— curated_train, minus near-duplicates, is the training set"),
    ], size=19)

    # 14b intermission (Stuttgart format: part one 6–7 PM, lightning talks, part two 7:30–8:30)
    s = d.slide("END OF PART ONE. The Nebius job was submitted during the embeddings segment; say so: "
                "'a GPU in Finland is training while we listen to the talks.' Repo QR stays up through the break.")
    d.kicker(s, "Part one done")
    d.text(s, Inches(0.9), Inches(1.7), Inches(11.5), Inches(1.4), "Lightning talks. Then we train.", size=54, color=INK, bold=True)
    d.bullets(s, [
        ("Right now on Nebius ", "— YOLO11n is fine-tuning on the frames we just labeled (L40S, 15 epochs, ~10 min)"),
        ("Part two ", "— pull the weights, evaluate, put detections back on the recording, then Nebius"),
        ("Catch up ", "— notebooks 01–04 are what we did; 05–06 are next"),
    ], y=Inches(3.4), size=21, w=Inches(7.6))
    d.qr(s, "repo", Inches(9.4), Inches(2.6), size=Inches(2.8), caption="github.com/danielgural/physical-ai-workshop")

    # 15 segment: train
    s = d.segment("train", "Fine-tune YOLO11n on Nebius", "75–95 min",
                  "Export to YOLO format. Stage on Nebius Object Storage. Submit a Serverless AI Job from a notebook. Pull best.pt.",
                  "SUBMIT THE JOB before this slide (during segment 4 Q&A). Show presenter/train_on_nebius.ipynb: export cell, "
                  "upload, submit → job id → Nebius console AI Jobs page → poll states. While it trains: the job spec slide. "
                  "If late, weights/best-dryrun.pt is the fallback and you say so.", hot=4)
    d.follow_along(s, "05_train_eval.ipynb  (predictions + eval baked in)", "Nebius Serverless AI Jobs · L40S · yolo11n.pt · 15 epochs")

    # 16 job spec
    s = d.slide("This is the whole job. No orchestrator, no plugin: SDK call, container, bucket.")
    d.kicker(s, "Train", color=NEBIUS)
    d.title(s, "A training job is a container, a bucket and one API call")
    d.code(s, Inches(0.9), Inches(2.3), Inches(7.4), Inches(3.7), [
        "from presenter.nebius_job import NebiusJob",
        "job = NebiusJob()                          # IAM token + Object Storage key",
        'job.upload_dir("build/yolo_export", "runs/stuttgart/input")',
        "",
        'job_id = job.submit("stuttgart", "runs/stuttgart",',
        '    model="yolo11n.pt", epochs=15, imgsz=640,',
        '    platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb")',
        "",
        "job.wait(job_id)   # PROVISIONING → RUNNING → COMPLETED",
        'job.download("runs/stuttgart/output/best.pt", "weights/best.pt")',
    ], size=14)
    d.bullets(s, [
        ("Serverless ", "— no cluster; a GPU appears for the job and goes away"),
        ("Image ", "— ghcr.io/danielgural/fiftyone-yolo-train, public"),
        ("Contract ", "— INPUT_S3_URI · OUTPUT_S3_URI · MODEL · HYPERPARAMS_JSON"),
        ("Data ", "— stays in Nebius Object Storage, eu-north1"),
    ], y=Inches(2.3), size=15, w=Inches(4.6))
    s.shapes[-1].left = Inches(8.6)

    # 17 evaluate
    s = d.slide("Model Evaluation panel: click the confusion-matrix cells. Then sort val by eval_yolo_fp. "
                "Then the missed grippers — what does the model not see, and which episodes.")
    d.kicker(s, "Evaluate")
    d.title(s, "Evaluate against the auto-labels — every cell is a view")
    d.bullets(s, [
        ("evaluate_detections ", "— mAP, per-class P/R, confusion matrix; results live on the samples as eval_yolo_tp / fp / fn"),
        ("Click a confusion cell ", "— the grid shows exactly those frames"),
        ("Worst-first ", "— sort validation by false positives; open the episodes"),
        ("Missed grippers ", "— false negatives grouped by episode tell you what to collect next"),
    ], size=19, w=Inches(7))
    d.code(s, Inches(8.2), Inches(2.3), Inches(4.3), Inches(2.6), [
        "results = val.evaluate_detections(",
        '    "yolo11n_preds",',
        '    gt_field="auto_labels",',
        '    eval_key="eval_yolo",',
        "    compute_mAP=True)",
        "results.print_report()",
    ], size=13)

    # 18 segment: close the loop
    s = d.segment("evaluate", "Close the loop on the timeline", "95–105 min",
                  "Detections on frames become temporal tags on the recording. Query the episodes by what the model saw.",
                  "06 notebook: intervals_from_predictions → brick-visible tags → open an episode: grasp / release / brick-visible "
                  "rows on the timeline. Rank episodes by mean gripper confidence. Then the flywheel sentence.", hot=[0, 5])
    d.follow_along(s, "06_close_the_loop.ipynb", "demo.fiftyone.ai · predictions back on Droid demo")

    # 19 the loop, closed
    s = d.slide("Payoff. Every step mapped to what a production team does. Flywheel: the next collection run "
                "gets curated with the views and tags saved today.")
    d.kicker(s, "Wrap")
    d.loop_row(s, Inches(1.0), hot=list(range(6)))
    d.text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(0.7), "Same loop. Bigger plumbing.", size=32, color=INK, bold=True)
    rows = [
        ("today, your laptop", "a production robotics team"),
        ("10 episodes, 5k frames", "millions of episodes in a data lake"),
        ("you scrubbing an episode", "triage teams on every collection run"),
        ("CLIP baked into the dataset", "embedding jobs on Nebius on every ingest"),
        ("Grounding DINO labels", "VLM labelers on Token Factory, human verify queues"),
        ("one YOLO11n on an L40S", "scheduled fine-tunes gated by curation queries"),
        ("brick-visible on ten timelines", "model output as a first-class field on every recording"),
    ]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.9), Inches(2.8), Inches(11.5), Inches(3.2)).table
    tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = Inches(4.3), Inches(0.6), Inches(6.6)
    tbl.first_row = False
    tbl.horz_banding = False
    for i, (a, b) in enumerate(rows):
        for j, (t, c) in enumerate(((a, ACCENT2), ("→", DIM), (b, INK))):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG if i % 2 == 0 else PANEL
            cell.margin_top = cell.margin_bottom = Emu(40000)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = t
            r.font.size = Pt(14)
            r.font.name = "Menlo"
            r.font.color.rgb = c

    # 20 do this on your data
    s = d.slide("Two CTAs, in this order: FiftyOne on your own recordings (self-serve QR), then Nebius hands over.")
    d.kicker(s, "Your data")
    d.title(s, "Do this on your recordings this week")
    d.bullets(s, [
        ("pip install fiftyone ", "— drop one of your own .mcap files into the MCAP Explorer. Watch 20 episodes. You will find something."),
        ("Hosted FiftyOne ", "— app.voxel51.com: same App, no install, datasets and brain runs in the cloud"),
        ("Take the repo home ", "— extract_frames.py, the notebooks, the Nebius job helper are all yours"),
    ], size=19, w=Inches(7.6))
    d.qr(s, "selfserve", Inches(9.4), Inches(2.3), size=Inches(2.8), caption="app.voxel51.com")

    # 21 Nebius handoff
    s = d.slide("Hand over to Nebius. If Nebius has no speaker at this stop, present 22–25 yourself; content is theirs.")
    d.kicker(s, "Nebius", color=NEBIUS)
    d.text(s, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.4), "What else runs on Nebius", size=54, color=INK, bold=True)
    d.text(s, Inches(0.9), Inches(3.9), Inches(11.5), Inches(1.0),
           "The GPU you just watched train a detector is one of three surfaces built for the Physical AI loop.", size=22, color=DIM)

    # 22 three activation blocks
    s = d.slide("From the Nebius H2 plan. Serverless is what we used. Token Factory is where the labeler VLMs live.")
    d.kicker(s, "Nebius", color=NEBIUS)
    d.title(s, "Three surfaces for the Physical AI loop")
    cols = [
        ("Workbench CLI", "Build, train and deploy Physical AI systems from one open-source CLI — containerized tools, SkyPilot workflows, SDK entry points."),
        ("Serverless AI Jobs", "Provision in under three minutes for simulation, synthetic data and policy training. Pay for the job, not a cluster. What you saw today."),
        ("Token Factory", "Open-weight chat and vision models behind one API — Qwen, GLM, Kimi — for labeling, review and augmentation of real-world data."),
    ]
    for i, (h, body) in enumerate(cols):
        x = Inches(0.9) + Inches(3.9) * i
        d.panel(s, x, Inches(2.3), Inches(3.6), Inches(3.6))
        d.text(s, x + Inches(0.25), Inches(2.5), Inches(3.1), Inches(0.6), h, size=22, color=NEBIUS, bold=True)
        d.text(s, x + Inches(0.25), Inches(3.2), Inches(3.1), Inches(2.6), body, size=16, color=INK)

    # 23 token factory for this loop
    s = d.slide("Concrete: where Token Factory plugs into the loop we just ran.")
    d.kicker(s, "Nebius", color=NEBIUS)
    d.title(s, "Where Token Factory plugs into today's loop")
    d.bullets(s, [
        ("Label ", "— a vision-language model reads the frame and the task string: 'is the brick in the drawer?' becomes a field"),
        ("Review ", "— the low-confidence queue goes to a VLM first, humans second"),
        ("Describe ", "— per-episode natural-language summaries for search: 'episodes where the cloth slipped'"),
        ("OpenAI-compatible API ", "— api.tokenfactory.nebius.com/v1; the same code you already write"),
        ("Credits for builders ", "— through the Builder Program, next slide"),
    ], size=19)

    # 24 office hours
    s = d.slide("Nebius's recurring format. If they announced dates, add them here.")
    d.kicker(s, "Nebius", color=NEBIUS)
    d.title(s, "Keep going: Physical AI Office Hours")
    d.bullets(s, [
        ("Biweekly online sessions ", "— raw robotics data to a working model in one sitting"),
        ("Voxel51 workshop series ", "— this format, 15 cities, September–November 2026"),
        ("Builders & Brews ", "— casual monthly builds; Berlin, Amsterdam, Paris, London, Warsaw…"),
    ], size=20)

    # 25 builder program CTA
    s = d.slide(f"THE CTA. QR goes to link.voxel51.com/{city['qr']} which prefills the form with this event. "
                "Ask everyone to scan now, not later; it takes name, email, company, GitHub. Goal is 500 sign-ups across the roadshow.")
    d.kicker(s, "Nebius Builder Program", color=NEBIUS)
    d.title(s, "GPU credits for what you build next")
    d.bullets(s, [
        ("Scan now ", "— the form is prefilled with this workshop; it takes a minute"),
        ("Credits + onboarding ", "— from sign-up to first job with a human, not a doc dump"),
        ("Serverless, Workbench, Token Factory ", "— everything you saw today"),
        ("Bring your recordings ", "— the repo's job helper works on your bucket unchanged"),
    ], size=19, w=Inches(7.4))
    d.qr(s, city["qr"], Inches(9.2), Inches(2.1), size=Inches(3.2), caption=f"link.voxel51.com/{city['qr']}")

    # 26 thanks / links
    s = d.slide("Q&A. Leave this up.")
    d.text(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(1.2), "Look at your data.", size=56, color=INK, bold=True)
    d.text(s, Inches(0.9), Inches(3.0), Inches(7.5), Inches(2.6), [
        [("repo        ", ACCENT, True), ("github.com/danielgural/physical-ai-workshop", INK, False)],
        [("datasets    ", ACCENT, True), ("huggingface.co/datasets/dgural/droid-mcap-workshop", INK, False)],
        [("            ", ACCENT, True), ("huggingface.co/datasets/dgural/droid-frames-workshop", INK, False)],
        [("fiftyone    ", ACCENT, True), ("github.com/voxel51/fiftyone  ·  app.voxel51.com", INK, False)],
        [("nebius      ", NEBIUS, True), (f"link.voxel51.com/{city['qr']}", INK, False)],
    ], size=17, font="Menlo")
    d.qr(s, "repo", Inches(8.9), Inches(3.0), size=Inches(1.9), caption="repo")
    d.qr(s, city["qr"], Inches(11.0), Inches(3.0), size=Inches(1.9), caption="nebius")
    d.text(s, Inches(0.9), Inches(6.2), Inches(11), Inches(0.5),
           "DROID: Khazatsky et al. 2024, CC-BY-4.0 · droid-dataset.github.io", size=13, color=DIM)

    out = HERE / f"physical-ai-workshop-{city_key}.pptx"
    d.prs.save(out)
    print(f"{out.name}: {d.n} slides")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="stuttgart", choices=list(CITIES))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for key in (CITIES if a.all else [a.city]):
        build(key)
