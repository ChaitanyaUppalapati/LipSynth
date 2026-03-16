# LipSynth

LipSynth is a research codebase for building and evaluating models that can infer spoken content from visual speech cues (lip movements) when the audio is missing or silent. The core goal is to enable silent speech detection and lip-reading by combining facial tracking, lip region extraction, and temporal modeling of mouth motion.

## Data Collection Pipeline

Our data collection pipeline starts from raw videos and runs a sequence of processing stages: we first segment videos into manageable clips, then apply face detection/recognition to locate subjects and crop the face region, and finally extract lip regions of interest (ROIs) across time. The pipeline also generates aligned transcripts and audio features so that models can be trained to predict audible speech content from the visual-only lip motion data. 