from __future__ import annotations

import argparse
from pathlib import Path

from workflow import BatteryStrategyService


def run(query: str, output_path: str | None, stream: bool) -> None:
    service = BatteryStrategyService()
    if stream:
        for event in service.stream(query):
            print(event)
    result = service.run(query=query)
    report = result.get("final_report", "")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"saved: {path}")
    else:
        print(report)
    print(result.get("criteria", {}))
    print(f"retry_count={result.get('retry_count', 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교해줘",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(query=args.query, output_path=args.output, stream=args.stream)
