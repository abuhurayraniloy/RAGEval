"""
evals/test_rate_limit.py

Fires 61 rapid requests at /search using a single API key and verifies
that the 61st request is rejected with 429 + Retry-After header.

Usage:
    python evals/test_rate_limit.py --base-url http://localhost:8000 --api-key <key>
"""

import argparse
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description="Test rate limiting")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key}
    payload = {"query": "rate limit test", "top_k": 1}

    statuses = []
    with httpx.Client(timeout=30.0) as client:
        for i in range(61):
            resp = client.post(f"{args.base_url}/search", json=payload, headers=headers)
            statuses.append(resp.status_code)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                print(f"Request {i + 1}: 429 (Retry-After: {retry_after})")
            else:
                print(f"Request {i + 1}: {resp.status_code}")

    ok_count = sum(1 for s in statuses if s == 200)
    rejected_count = sum(1 for s in statuses if s == 429)

    print(f"\n{ok_count} succeeded, {rejected_count} rate-limited")

    if len(statuses) == 61 and statuses[60] == 429:
        print("PASS: 61st request was correctly rate-limited.")
    else:
        print("FAIL: 61st request was not rate-limited as expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
