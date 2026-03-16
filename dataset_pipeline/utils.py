"""Common utilities for dataset pipeline."""

import subprocess
import os
import requests
from pathlib import Path

def load_env():
    """Load API keys from .env file."""
    for path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v.strip())

def get_playlist_video_id(playlist_id: str, index: int) -> str:
    """Get video ID from playlist at given index."""
    cmd = ["yt-dlp", "--playlist-items", str(index), "--print", "%(display_id)s",
           "--no-download", f"https://www.youtube.com/playlist?list={playlist_id}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip().split("\n")[0] if result.stdout else None

def get_video_description(video_id: str) -> str:
    """Get video description."""
    cmd = ["yt-dlp", "--print", "%(description)s", "--no-download", 
           f"https://www.youtube.com/watch?v={video_id}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def query_openrouter(prompt: str, model: str = "openai/gpt-3.5-turbo") -> str:
    """Query OpenRouter LLM API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/shra012/LipSynth",
            "X-Title": "LipSynth"
        },
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000},
        timeout=120
    )
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None

def parse_csv_lines(text: str) -> dict:
    """Parse CSV-like text into dict {index: value}."""
    result = {}
    for line in text.strip().split("\n"):
        if "," in line:
            parts = line.split(",", 1)
            try:
                result[int(parts[0].strip())] = parts[1].strip()
            except ValueError:
                continue
    return result
