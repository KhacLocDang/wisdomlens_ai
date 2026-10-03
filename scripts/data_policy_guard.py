from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
PRIVATE_DIR = DATA_ROOT / "private"
PUBLIC_DIR = DATA_ROOT / "public"
AUDIO_DIR = DATA_ROOT / "audio"


def ensure_data_layout() -> dict[str, Path]:
    """Create the recommended data layout for private/runtime data, public demo data, and audio files."""
    DATA_ROOT.mkdir(exist_ok=True)
    for directory in (PRIVATE_DIR, PUBLIC_DIR, AUDIO_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": DATA_ROOT,
        "private": PRIVATE_DIR,
        "public": PUBLIC_DIR,
        "audio": AUDIO_DIR,
    }


def print_layout_summary() -> None:
    layout = ensure_data_layout()
    print("Recommended data layout:")
    for name, path in layout.items():
        print(f"- {name}: {path.relative_to(ROOT)}")

    print("\nRules:")
    print("1. Keep raw DB dumps, production data, and private exports under data/private/")
    print("2. Keep only sanitized/public-safe sample data under data/public/")
    print("3. Keep generated audio files under data/audio/")
    print("4. Never push raw production exports or personal data to GitHub.")


if __name__ == "__main__":
    print_layout_summary()
