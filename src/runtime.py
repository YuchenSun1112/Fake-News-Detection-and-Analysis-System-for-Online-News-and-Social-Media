import os
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / ".tmp"
BLAT_RE = re.compile(r"^[A-Za-z0-9_]{8}$")


def configure_temp_dir() -> Path:
    TEMP_DIR.mkdir(exist_ok=True)
    tempfile.tempdir = str(TEMP_DIR)
    os.environ.setdefault("TEMP", str(TEMP_DIR))
    os.environ.setdefault("TMP", str(TEMP_DIR))
    return TEMP_DIR


def cleanup_blat_temp_files() -> int:
    removed = 0
    for directory in (PROJECT_ROOT, configure_temp_dir()):
        for path in directory.iterdir():
            if not path.is_file() or not BLAT_RE.fullmatch(path.name):
                continue
            try:
                if path.stat().st_size == 4:
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
