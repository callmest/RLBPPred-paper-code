#!/usr/bin/env python3
"""Replay the current manuscript's frozen quantitative figure panels.

The historical 342-candidate implementation is reproduce_figures_legacy.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.current_figures.replay import main

if __name__ == "__main__":
    main()
