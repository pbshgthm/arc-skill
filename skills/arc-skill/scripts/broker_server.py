#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from arc_skill.broker import serve
from arc_skill.core import RunPaths


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    serve(RunPaths(Path(args.run_dir).resolve()))


if __name__ == "__main__":
    main()
