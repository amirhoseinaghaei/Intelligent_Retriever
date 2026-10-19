#!/usr/bin/env python3
"""Standalone entry point for investigate data extraction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_posttraining.cli.extract import main

if __name__ == "__main__":
    main()
