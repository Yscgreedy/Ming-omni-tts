from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_CACHE_TTL_SECONDS = 1800
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
DEFAULT_LOCK_STALE_SECONDS = 120.0


class PromptAudioCacheError(RuntimeError):
    pass


def is_remote_prompt_audio(path: str | None) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def _get_cache_dir() -> Path:
    configured = os.getenv("PROMPT_AUDIO_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / "runtime-cache" / "prompt-audio"


def _get_cache_ttl_seconds() -> int:
    raw_value = os.getenv("PROMPT_AUDIO_CACHE_TTL_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_CACHE_TTL_SECONDS
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_CACHE_TTL_SECONDS
    return parsed if parsed > 0 else DEFAULT_CACHE_TTL_SECONDS


def _normalize_url(url: str) -> str:
    return url.strip()


def _build_cache_file_path(cache_dir: Path, url: str) -> Path:
    normalized_url = _normalize_url(url)
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
    parsed = urlparse(normalized_url)
    suffix = Path(parsed.path).suffix or ".audio"
    return cache_dir / f"{digest}{suffix}"


def _is_cache_file_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return (time.time() - path.stat().st_mtime) <= ttl_seconds


def _touch(path: Path) -> None:
    now = time.time()
    os.utime(path, (now, now))


def _cleanup_expired_files(cache_dir: Path, ttl_seconds: int) -> None:
    if not cache_dir.exists():
        return
    now = time.time()
    expiry_seconds = max(ttl_seconds, DEFAULT_LOCK_STALE_SECONDS)
    for entry in cache_dir.iterdir():
        if not entry.is_file():
            continue
        if (now - entry.stat().st_mtime) <= expiry_seconds:
            continue
        entry.unlink(missing_ok=True)


class _FileLock:
    def __init__(self, lock_path: Path, timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS) -> None:
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self._acquired = False

    def __enter__(self) -> "_FileLock":
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self.lock_path.exists() and (time.time() - self.lock_path.stat().st_mtime) > DEFAULT_LOCK_STALE_SECONDS:
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if time.time() >= deadline:
                    raise PromptAudioCacheError(f"Timed out waiting for prompt audio cache lock: {self.lock_path.name}")
                time.sleep(0.1)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(str(os.getpid()))
            self._acquired = True
            return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            self.lock_path.unlink(missing_ok=True)


def resolve_prompt_audio_path(path_or_url: str) -> str:
    if not is_remote_prompt_audio(path_or_url):
        return path_or_url

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ttl_seconds = _get_cache_ttl_seconds()
    _cleanup_expired_files(cache_dir, ttl_seconds)

    cache_path = _build_cache_file_path(cache_dir, path_or_url)
    if _is_cache_file_fresh(cache_path, ttl_seconds):
        _touch(cache_path)
        return str(cache_path)

    lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
    with _FileLock(lock_path):
        if _is_cache_file_fresh(cache_path, ttl_seconds):
            _touch(cache_path)
            return str(cache_path)

        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.part.{os.getpid()}")
        try:
            response = requests.get(path_or_url, timeout=30)
            response.raise_for_status()
            temp_path.write_bytes(response.content)
            if temp_path.stat().st_size == 0:
                raise PromptAudioCacheError(f"Downloaded prompt audio is empty: {path_or_url}")
            temp_path.replace(cache_path)
            _touch(cache_path)
            return str(cache_path)
        except requests.RequestException as exc:
            raise PromptAudioCacheError(f"Failed to download prompt audio: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)
