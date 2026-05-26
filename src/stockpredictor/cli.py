from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .backtesting import run_backtest
from .config import load_settings
from .pipeline import analyze_symbol, scan_symbols
from .utils import to_serializable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpredictor")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api", parents=[common], help="Run the FastAPI service.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    dashboard_parser = subparsers.add_parser("dashboard", parents=[common], help="Run the Streamlit dashboard.")
    dashboard_parser.add_argument("--server-port", type=int, default=8501)

    analyze_parser = subparsers.add_parser("analyze", parents=[common], help="Analyze one symbol.")
    analyze_parser.add_argument("symbol")

    scan_parser = subparsers.add_parser("scan", parents=[common], help="Scan a watchlist or symbols.")
    scan_parser.add_argument("symbols", nargs="*")

    backtest_parser = subparsers.add_parser("backtest", parents=[common], help="Run the configured backtest.")
    backtest_parser.add_argument("symbols", nargs="*")

    args = parser.parse_args(argv)

    if args.command == "api":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(args.config), host=args.host, port=args.port)
        return 0

    if args.command == "dashboard":
        dashboard_path = Path(__file__).resolve().parent / "ui" / "dashboard.py"
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(args.server_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--",
            "--config",
            args.config,
        ]
        return subprocess.call(command)

    settings = load_settings(args.config)
    if args.command == "analyze":
        _print_json(analyze_symbol(args.symbol, settings))
    elif args.command == "scan":
        _print_json(scan_symbols(settings, symbols=args.symbols or None))
    elif args.command == "backtest":
        _print_json(run_backtest(settings, symbols=args.symbols or None))
    return 0


def _print_json(value: object) -> None:
    print(json.dumps(to_serializable(value), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
