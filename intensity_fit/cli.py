from __future__ import annotations

import argparse

from .workflow import run_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="intensity-fit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run an intensity fit from a JSON/YAML config")
    run.add_argument("config")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_from_config(args.config)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
