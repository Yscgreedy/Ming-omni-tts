from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import prompt_audio_cache


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_remote_prompt_audio_is_cached(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setenv("PROMPT_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PROMPT_AUDIO_CACHE_TTL_SECONDS", "1800")

    def fake_get(url: str, timeout: float):  # noqa: ARG001
        calls.append(url)
        return FakeResponse(b"RIFF-cache")

    monkeypatch.setattr(prompt_audio_cache.requests, "get", fake_get)

    first_path = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=1"))
    second_path = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=1"))

    assert first_path == second_path
    assert first_path.read_bytes() == b"RIFF-cache"
    assert calls == ["https://backend.example.com/audio.wav?v=1"]


def test_remote_prompt_audio_uses_full_url_for_cache_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_AUDIO_CACHE_DIR", str(tmp_path))

    def fake_get(url: str, timeout: float):  # noqa: ARG001
        return FakeResponse(url.encode("utf-8"))

    monkeypatch.setattr(prompt_audio_cache.requests, "get", fake_get)

    first_path = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=1"))
    second_path = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=2"))

    assert first_path != second_path
    assert first_path.read_bytes() != second_path.read_bytes()


def test_remote_prompt_audio_redownloads_after_ttl(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    now = [1000.0]

    monkeypatch.setenv("PROMPT_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PROMPT_AUDIO_CACHE_TTL_SECONDS", "10")
    monkeypatch.setattr(prompt_audio_cache.time, "time", lambda: now[0])

    def fake_get(url: str, timeout: float):  # noqa: ARG001
        calls.append(url)
        return FakeResponse(f"content-{len(calls)}".encode("utf-8"))

    monkeypatch.setattr(prompt_audio_cache.requests, "get", fake_get)

    cached_path = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=1"))
    assert cached_path.read_bytes() == b"content-1"

    now[0] += 11
    cached_path_after_expiry = Path(prompt_audio_cache.resolve_prompt_audio_path("https://backend.example.com/audio.wav?v=1"))
    assert cached_path_after_expiry == cached_path
    assert cached_path_after_expiry.read_bytes() == b"content-2"
    assert calls == [
        "https://backend.example.com/audio.wav?v=1",
        "https://backend.example.com/audio.wav?v=1",
    ]
