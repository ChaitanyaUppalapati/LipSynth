import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from functools import wraps


def _to_jsonable(value: Any) -> Any:
    """Recursively convert non-JSON-native values (e.g. Path) into JSON-safe types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class PipelineStepState:
    step_name: str
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    items_processed: int = 0
    items_total: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class PipelineState:
    pipeline_name: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    current_step: int = 0
    steps: list = field(default_factory=list)
    input_config: dict = field(default_factory=dict)
    output_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        parsed = dict(data)
        parsed_steps = []
        step_fields = set(PipelineStepState.__dataclass_fields__.keys())

        for step in parsed.get("steps", []):
            if isinstance(step, PipelineStepState):
                parsed_steps.append(step)
                continue

            if isinstance(step, dict):
                # Backward compatibility: ignore unknown keys in older/newer checkpoint schemas.
                filtered = {k: v for k, v in step.items() if k in step_fields}
                parsed_steps.append(PipelineStepState(**filtered))

        parsed["steps"] = parsed_steps
        return cls(**parsed)

class CheckpointManager:
    def __init__(self, pipeline_name: str, checkpoint_dir: str = "checkpoints"):
        self.pipeline_name = pipeline_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{pipeline_name}_checkpoint.json"
        self._state: Optional[PipelineState] = None

    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()

    def init_state(self, input_config: dict = None, output_config: dict = None) -> PipelineState:
        state = PipelineState(
            pipeline_name=self.pipeline_name,
            input_config=input_config or {},
            output_config=output_config or {},
            created_at=self._get_timestamp(),
            last_updated=self._get_timestamp()
        )
        self._state = state
        self._save_state(state)
        return state

    def load(self) -> Optional[PipelineState]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                self._state = PipelineState.from_dict(data)
                return self._state
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                return None
        return None

    def _save_state(self, state: PipelineState):
        state.last_updated = self._get_timestamp()
        with open(self.checkpoint_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def save(self, step_name: str = None, status: str = None, step_data: dict = None, force: bool = False):
        if self._state is None:
            self.load()
        if self._state is None:
            self.init_state()
        if step_name:
            step_found = False
            for step in self._state.steps:
                if step.step_name == step_name:
                    if status:
                        step.status = status
                    if step_data:
                        step.metadata.update(step_data)
                    if status == "running" and step.started_at is None:
                        step.started_at = self._get_timestamp()
                    if status == "completed" and step.completed_at is None:
                        step.completed_at = self._get_timestamp()
                    step_found = True
                    break
            if not step_found:
                new_step = PipelineStepState(
                    step_name=step_name,
                    status=status or "running",
                    started_at=self._get_timestamp() if status == "running" else None
                )
                if step_data:
                    new_step.metadata = step_data
                self._state.steps.append(new_step)
            for i, step in enumerate(self._state.steps):
                if step.step_name == step_name:
                    self._state.current_step = i
                    break
        self._save_state(self._state)

    def get_step_state(self, step_name: str) -> Optional[PipelineStepState]:
        if self._state is None:
            self.load()
        if self._state is None:
            return None
        for step in self._state.steps:
            if step.step_name == step_name:
                return step
        return None

    def is_step_completed(self, step_name: str) -> bool:
        step = self.get_step_state(step_name)
        return step is not None and step.status == "completed"

    def get_completed_steps(self) -> list:
        if self._state is None:
            self.load()
        if self._state is None:
            return []
        return [s.step_name for s in self._state.steps if s.status == "completed"]

    def reset_step(self, step_name: str):
        self.save(step_name=step_name, status="pending")

    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._state = None

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0,
                       max_delay: float = 60.0, exceptions: tuple = (Exception,), on_retry: Optional[Callable] = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt >= max_retries:
                        print(f"[ERROR] {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    print(f"[RETRY] {func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                    if on_retry:
                        on_retry(attempt, e, delay)
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

class ProgressTracker:
    def __init__(self, total: int = 0, name: str = "Progress"):
        self.total = total
        self.name = name
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()

    def update(self, success: bool = True, skipped: bool = False):
        self.processed += 1
        if skipped:
            self.skipped += 1
        elif success:
            self.succeeded += 1
        else:
            self.failed += 1

    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.processed / max(elapsed, 0.001)

    def get_eta(self) -> float:
        if self.processed == 0 or self.total == 0:
            return 0
        rate = self.get_rate()
        remaining = self.total - self.processed
        return remaining / max(rate, 0.001)

    def get_progress_pct(self) -> float:
        if self.total == 0:
            return 0
        return (self.processed / self.total) * 100

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def get_status(self) -> str:
        pct = self.get_progress_pct()
        rate = self.get_rate()
        eta = self.get_eta()
        status = f"{self.name}: {self.processed}/{self.total} ({pct:.1f}%) | {rate:.1f}/s | "
        status += f"ETA: {self.format_time(eta)}" if eta > 0 else f"Done in {self.format_time(time.time() - self.start_time)}"
        if self.failed > 0:
            status += f" | {self.failed} failed"
        if self.skipped > 0:
            status += f" | {self.skipped} skipped"
        return status

    def print_progress(self):
        print(f"\r{self.get_status()}", end="", flush=True)

    def get_summary(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "name": self.name,
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_seconds": round(elapsed, 2),
            "rate_per_second": round(self.get_rate(), 2),
            "success_rate": round((self.succeeded / max(self.processed, 1)) * 100, 1)
        }

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)

def is_file_complete(path: str, min_size_kb: int = 1) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > min_size_kb * 1024

def list_files_by_extension(directory: str, extension: str) -> list:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted([str(f) for f in dir_path.rglob(f"*.{extension.lstrip('.')}")])

def save_json(data: dict, path: str, indent: int = 2):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def update_json(path: str, update_func: Callable[[dict], dict]):
    data = {}
    if Path(path).exists():
        data = load_json(path)
    updated = update_func(data)
    save_json(updated, path)

def pipeline_step(step_name: str, checkpoint_manager: CheckpointManager = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if checkpoint_manager and checkpoint_manager.is_step_completed(step_name):
                print(f"[SKIP] {step_name} already completed, skipping...")
                return None
            if checkpoint_manager:
                checkpoint_manager.save(step_name=step_name, status="running")
            try:
                result = func(*args, **kwargs)
                if checkpoint_manager:
                    checkpoint_manager.save(step_name=step_name, status="completed")
                return result
            except Exception as e:
                if checkpoint_manager:
                    checkpoint_manager.save(step_name=step_name, status="failed",
                                           step_data={"error": str(e), "traceback": traceback.format_exc()})
                raise
        return wrapper
    return decorator

def get_pipeline_state(checkpoint_dir: str = "checkpoints") -> Optional[PipelineState]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    checkpoints = list(checkpoint_path.glob("*_checkpoint.json"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    with open(latest, "r") as f:
        return PipelineState.from_dict(json.load(f))
