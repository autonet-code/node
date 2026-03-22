"""
Autonet Node Entry Point.

Run a solver node as a managed background service:

    python -m nodes
    python -m nodes --config path/to/autonet.yaml

"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Autonet Node — Decentralized AI Training Service",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to autonet.yaml config file",
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args()

    if args.version:
        try:
            from nodes import __version__
            print(f"autonet {__version__}")
        except (ImportError, AttributeError):
            print("autonet 0.1.0")
        sys.exit(0)

    from .common.config import load_config
    from .service import AutonetService

    config = load_config(args.config)
    service = AutonetService(config)
    service.start()


if __name__ == "__main__":
    main()
