"""CLI entry point for HEL1OS multi-orbit concatenation.

Usage:
    python -m bah2026.data.hel1os_concat [--date YYYY-MM-DD] [--check-only]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the HEL1OS multi-orbit concatenation from the CLI."""
    # Import from the script in data/downloads/
    script_path = (
        Path(__file__).resolve().parents[3] / "data" / "downloads" / "concat_orbits.py"
    )
    sys.path.insert(0, str(script_path.parent))

    import importlib

    spec = importlib.util.spec_from_file_location("concat_orbits", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
