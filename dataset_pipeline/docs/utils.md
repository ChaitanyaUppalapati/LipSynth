# Dataset Pipeline Utils

Common utilities for the LipSynth dataset pipeline.

## Functions

### load_env()
Load API keys from `.env` file. Checks both `dataset_pipeline/.env` and project root `.env`.

### get_playlist_video_id(playlist_id: str, index: int) -> str
Get YouTube video ID from playlist at given index.
- `playlist_id`: YouTube playlist ID
- `index`: Position in playlist (1-based)

### get_video_description(video_id: str) -> str
Get video description using yt-dlp.

### query_openrouter(prompt: str, model: str = "openai/gpt-3.5-turbo") -> str
Query OpenRouter LLM API. Returns response text or None if failed.
- Requires `OPENROUTER_API_KEY` in environment or .env file

### parse_csv_lines(text: str) -> dict
Parse CSV-like text into dict `{index: value}`.

## .env File

Create `dataset_pipeline/.env` with:
```
OPENROUTER_API_KEY=your_key_here
```

Get free key at https://openrouter.ai
