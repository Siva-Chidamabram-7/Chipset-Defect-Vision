#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one or more images to the inference API and record latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/predict",
        help="Predict endpoint URL.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths to test.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def post_base64_image(url: str, image_path: Path, timeout: int) -> tuple[float, dict]:
    started_at = time.perf_counter()
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = urllib.parse.urlencode({"image_base64": image_base64}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        body = json.loads(response.read().decode("utf-8"))
        return elapsed_ms, body


def main() -> int:
    args = parse_args()
    latencies: list[float] = []

    print("=" * 72)
    print("Inference performance check")
    print(f"Endpoint : {args.url}")
    print(f"Images   : {len(args.images)}")
    print("=" * 72)

    for raw_path in args.images:
        image_path = Path(raw_path).expanduser().resolve()
        if not image_path.exists():
            print(f"[ERROR] Missing image: {image_path}")
            return 1

        try:
            latency_ms, response = post_base64_image(args.url, image_path, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[ERROR] {image_path.name}: HTTP {exc.code} {detail}")
            return 1
        except Exception as exc:  # pragma: no cover
            print(f"[ERROR] {image_path.name}: {exc}")
            return 1

        latencies.append(latency_ms)
        summary = response.get("summary", {})
        print(
            f"{image_path.name}: "
            f"status={response.get('status')} "
            f"defects={summary.get('defect_count', 0)} "
            f"total={summary.get('total', 0)} "
            f"latency_ms={latency_ms:.2f}"
        )

    average_ms = sum(latencies) / len(latencies)
    print("-" * 72)
    print(f"Average latency: {average_ms:.2f} ms")
    print(f"Min latency    : {min(latencies):.2f} ms")
    print(f"Max latency    : {max(latencies):.2f} ms")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
