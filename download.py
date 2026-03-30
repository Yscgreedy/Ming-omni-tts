import os
import sys
import importlib
from pathlib import Path


REPO_ID = os.getenv("MODEL_REPO_ID", "inclusionAI/Ming-omni-tts-0.5B")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.join("Models", "hub"))
LOCK_FILE = os.path.join(CACHE_DIR, ".model_download.lock")


class FileLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self.handle = open(self.lock_path, "a+")

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is None:
            return False

        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)

        self.handle.close()
        return False


def model_already_cached(repo_id: str, cache_dir: str) -> bool:
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download

    try:
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


def main() -> int:
    if os.path.exists("/Models"):
        return 0
    try:
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    except ImportError as exc:
        print(
            "[download] missing dependency: huggingface_hub. Please run `pip install -r requirements.txt` first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"[download] repo={REPO_ID}")
    print(f"[download] cache_dir={cache_path.resolve()}")

    with FileLock(LOCK_FILE):
        if model_already_cached(REPO_ID, CACHE_DIR):
            print("[download] model already cached, skip download")
            return 0

        print("[download] model not found in cache, downloading...")
        local_snapshot = snapshot_download(repo_id=REPO_ID, cache_dir=CACHE_DIR)
        print(f"[download] completed: {local_snapshot}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[download] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
