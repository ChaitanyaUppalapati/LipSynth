# Lip Reading Dataset ETL Pipeline
## Build Your Own LipVoicer-Compatible Dataset from YouTube

---

## Overview

This pipeline builds a sentence-level audiovisual speech dataset from YouTube videos,
structured to be compatible with LipVoicer training. It handles everything from download
to train/val/test split creation.

```
YouTube URLs → Download → Transcribe → Segment → Face Detect → Lip Crop → Dataset
     (CSV)       (yt-dlp)  (Whisper)   (ffmpeg)  (PyTorch)    (OpenCV)   (manifests)
```

---

## Dataset Specifications

| Parameter              | Our Target         | LRS2 Reference     |
|------------------------|--------------------|---------------------|
| Speakers               | 50                 | 1000+               |
| Hours total            | 15-25h             | 224h                |
| Minutes per speaker    | 15-30 min          | ~13 min average     |
| Clip duration          | 2-10 seconds       | ≤10 seconds         |
| Max characters/clip    | 100                | 100                 |
| Video FPS              | 25                 | 25                  |
| Video resolution       | 720p               | varies              |
| Audio sample rate      | 16kHz mono         | 16kHz               |
| Mouth ROI size         | 96×96 grayscale    | 96×96 grayscale     |
| Vocabulary             | open               | open                |
| Train/Val/Test split   | 80/10/10 (by spk)  | by broadcast date   |

### How Much Video Per Speaker?

**Target: 15-30 minutes of usable speech per speaker.**

Here's the math:
- A typical 20-minute YouTube talk has ~12-15 min of actual clear, single-speaker speech
  (rest is applause, title cards, audience shots, Q&A with off-camera audio)
- From 15 min of speech, you'll get ~100-150 valid clips (after quality filtering)
- That's about 12-14 min of final segmented data per speaker
- 50 speakers × ~15 min each = ~12.5 hours of training data

**So aim for 20-40 min source videos per speaker.** A single TED talk or long interview works great.

If a speaker only has short videos (5-10 min), you can use 2-3 videos per speaker:
```csv
spk_001,https://youtube.com/watch?v=AAA,Speaker_1
spk_001,https://youtube.com/watch?v=BBB,Speaker_1
```
The pipeline will concatenate them.

---

## Video Selection Guide

### Best Sources (in order of quality for lip reading)

1. **TED / TEDx Talks** — Clear frontal face, good lighting, one speaker
2. **University lectures** — Consistent framing, minimal cuts
3. **Interview podcasts (video)** — Long single-speaker segments
4. **Press conferences** — Formal, clear speech
5. **YouTube educational channels** — Talking head format

### What to Look For

**Good for lip reading:**
- Speaker facing camera most of the time
- Good lighting on the face
- Minimal head movement / turning away
- Clear English speech at normal pace
- Minimal background noise
- At least 720p resolution
- Single speaker visible (not a panel/debate)

**Avoid:**
- Music videos, movie clips (copyright + editing)
- Multi-person debates / roundtables
- Heavy makeup tutorials (occlude lip area)
- Videos with lots of B-roll / cutaway shots
- Low resolution (<480p)
- Heavy accents (for initial training; add later for robustness)
- Significant face occlusion (masks, hands on face, microphones blocking mouth)

### Speaker Diversity Goals

For 50 speakers, aim for:
- ~25 male, ~25 female
- Age range: 20s through 60s
- Mix of accents (but all clearly intelligible English)
- Variety of skin tones
- Some with glasses, some without (adds robustness)

---

## Quick Start

### 1. Install dependencies
```bash
# from repo root:
pip install -e .
# ffmpeg is bundled via imageio-ffmpeg
```

### 2. Prepare your video list
Create `data/links/video_links.csv` and fill in:
```csv
speaker_id,youtube_url,speaker_name
spk_001,https://www.youtube.com/watch?v=XXXXX,John_Doe
spk_002,https://www.youtube.com/watch?v=YYYYY,Jane_Smith
...
```

### 3. Run the full pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh data/links/video_links.csv base pytorch
```

Or run steps individually:
```bash
# Step 0: Fetch/refresh playlist CSV
python 00_fetch_playlist.py --num_videos 3 --output data/links/video_links.csv

# Step 1: Download
python 01_download_videos.py --input data/links/video_links.csv --output_dir data/raw_videos/

# Step 2: Segment + Transcribe
python 02_segment_clips.py --input_dir data/raw_videos/ --output_dir data/segments/ --whisper_model base

# Step 3: Extract lip ROIs
python 03_extract_lip_roi.py --input_dir data/segments/ --output_dir data/lip_rois/ --detector pytorch

# Step 4: Finalize
python 04_finalize_dataset.py --segments_dir data/segments/ --lip_rois_dir data/lip_rois/ --output_dir data/dataset_final/
```

---

## Pipeline Steps in Detail

### Step 0: Fetch Playlist (`00_fetch_playlist.py`)
- Fetches playlist items and updates `data/links/video_links.csv`
- Uses LLM-assisted speaker naming when available
- Re-run safe: refreshes CSV to requested number of videos

### Step 1: Download (`01_download_videos.py`)
- Downloads each YouTube URL at 720p using yt-dlp
- Extracts audio as 16kHz mono WAV
- Skips already-downloaded videos (safe to re-run)

### Step 2: Segment (`02_segment_clips.py`)
- Transcribes full audio using OpenAI Whisper (with word-level timestamps)
- Segments into 2-10 second clips at sentence boundaries
- Filters: ≤100 characters, ≥2 words per clip
- Splits long segments at punctuation boundaries
- Output: individual .mp4 + .wav + .txt + .json per clip

### Step 3: Lip ROI Extraction (`03_extract_lip_roi.py`)
- Detects faces and landmarks using a PyTorch face-alignment model
- Extracts 96×96 grayscale mouth ROI per frame
- Quality filters:
  - Rejects clips with <85% face detection rate
  - Rejects clips with >30% multi-face frames
- Output: .npz file per clip with shape (T, 96, 96)

### Step 4: Finalize (`04_finalize_dataset.py`)
- Cross-validates all pipeline outputs
- Creates **speaker-disjoint** train/val/test splits
- Generates TSV manifests
- Computes dataset statistics
- Organizes into clean directory structure

---

## Final Dataset Structure

```
dataset_final/
├── train.tsv              # clip_id  text  speaker_id  duration  num_words
├── val.tsv
├── test.tsv
├── dataset_stats.json     # full statistics
├── videos/                # all clip .mp4 files
│   ├── spk_001_0001.mp4
│   └── ...
├── audios/                # all clip .wav files (16kHz mono)
│   ├── spk_001_0001.wav
│   └── ...
├── mouths/                # all mouth ROI .npz files (T, 96, 96)
│   ├── spk_001_0001.npz
│   └── ...
└── transcripts/           # all .txt files
    ├── spk_001_0001.txt
    └── ...
```

---

## Connecting to LipVoicer

To use this dataset with LipVoicer's training code:

1. **Videos directory** → maps to `dataset.videos_dir`
2. **Audios directory** → maps to `dataset.audios_dir`  
3. **Mouths directory** → maps to `dataset.mouthrois_dir`
4. You'll still need LipVoicer's pretrained components:
   - Visual Speech Recognition model (lip reading network)
   - ASR model (for classifier guidance)
   - These can be initialized from pretrained checkpoints

---

## Whisper Model Size Guide

| Model   | VRAM  | Speed (1hr audio) | Accuracy | Recommended For          |
|---------|-------|--------------------|----------|--------------------------|
| tiny    | ~1 GB | ~5 min             | okay     | Quick testing only       |
| base    | ~1 GB | ~10 min            | good     | Default — good tradeoff  |
| small   | ~2 GB | ~20 min            | better   | If you have time         |
| medium  | ~5 GB | ~45 min            | great    | Production quality       |
| large   | ~10GB | ~90 min            | best     | Final dataset only       |

**Recommendation:** Use `base` for initial pipeline testing, then re-run Step 2 with 
`medium` or `large` for your final dataset.

---

## Estimated Processing Time (50 speakers)

| Step                | Time Estimate           | Bottleneck     |
|---------------------|-------------------------|----------------|
| Download            | 30-60 min               | Network        |
| Transcribe (base)   | 2-4 hours               | CPU/GPU        |
| Lip ROI extraction  | 4-8 hours               | CPU (GPU helps)|
| Finalization        | < 5 min                 | Disk I/O       |
| **Total**           | **~7-13 hours**         |                |

With a GPU, transcription is much faster (Whisper supports CUDA).
PyTorch face-alignment can run on CPU or GPU (`--device auto` picks the best available).

---

## Troubleshooting

**yt-dlp fails on some videos:**
- Some videos have download restrictions; skip them
- Update yt-dlp: `pip install -U yt-dlp`

**Whisper runs out of memory:**
- Use a smaller model: `--whisper_model tiny`
- Or process on GPU if available

**Low face detection rate:**
- Check video quality — below 480p often fails
- Use GPU if available: `--device cuda`
- Some videos have too many cutaway shots; replace them

**Too few clips after filtering:**
- Lower `MIN_FRAMES_RATIO` in Step 3 (default 0.85)
- Use longer source videos (30+ min)
- Check that speakers face the camera
