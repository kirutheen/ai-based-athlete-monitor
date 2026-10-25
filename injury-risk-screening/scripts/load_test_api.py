#!/usr/bin/env python3
"""
Real concurrency load test against the actual FastAPI backend
(src/api/main.py) -- measures genuine throughput and latency on THIS
machine, then uses that real measurement to honestly discuss what the
"100k simultaneous users" target from the abstract would actually
require.

============================= HONEST SCOPE =================================
This is not a simulation -- it starts the real uvicorn server, the real
SQLite-backed FastAPI app, and fires real concurrent HTTP requests at it
with a real async HTTP client, then reports the real requests/sec and
latency this sandbox's single process/single SQLite file can sustain.

What this demonstrates: the actual bottleneck profile of the CURRENT
architecture (single process, SQLite, in-memory risk-tracker dict) under
concurrent load, which is a legitimate, measured data point.

What this does NOT do:
  - It does not prove the system can or can't handle 100,000
    simultaneous real users -- that depends on hardware, network,
    database choice, and horizontal scaling, none of which a single
    sandboxed container can demonstrate. 100k concurrent users is a
    distributed-systems capacity-planning question, not something a
    single load-test run answers.
  - The numbers below are specific to this sandbox's CPU/IO
    characteristics and will differ on real deployment hardware --
    treat them as "here is the current architecture's bottleneck
    pattern," not "here is the exact production number."

At the end, this script prints an honest, reasoned discussion of what
would actually need to change (database, process model, state
management) to move from today's number toward a 100k-user target --
that discussion is architectural reasoning, not a benchmark result.
==============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = "http://127.0.0.1:{port}"


def start_server(port: int) -> subprocess.Popen:
    """Launches the real FastAPI app with uvicorn as a subprocess (a
    genuine running server, not an in-process TestClient shortcut)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    base_url = BASE_URL.format(port=port)
    for _ in range(50):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("Server did not start in time")


async def _hit_health(client: httpx.AsyncClient, base_url: str) -> float:
    start = time.perf_counter()
    resp = await client.get(f"{base_url}/health")
    resp.raise_for_status()
    return time.perf_counter() - start


async def _create_athlete(client: httpx.AsyncClient, base_url: str, index: int) -> float:
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/athletes",
        json={"display_name": f"LoadTest Athlete {index}", "sport": "soccer", "dominant_leg": "right"},
    )
    resp.raise_for_status()
    return time.perf_counter() - start


async def run_load_test(base_url: str, n_requests: int, concurrency: int, endpoint: str) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def worker(index: int, client: httpx.AsyncClient):
        async with semaphore:
            if endpoint == "health":
                latency = await _hit_health(client, base_url)
            else:
                latency = await _create_athlete(client, base_url, index)
            latencies.append(latency)

    async with httpx.AsyncClient(timeout=30.0) as client:
        wall_start = time.perf_counter()
        await asyncio.gather(*[worker(i, client) for i in range(n_requests)])
        wall_elapsed = time.perf_counter() - wall_start

    latencies.sort()
    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "wall_time_s": wall_elapsed,
        "requests_per_sec": n_requests / wall_elapsed,
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_p50_ms": latencies[len(latencies) // 2] * 1000,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] * 1000,
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)] * 1000,
        "latency_max_ms": latencies[-1] * 1000,
    }


def print_results(label: str, result: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"Requests:          {result['n_requests']} (concurrency {result['concurrency']})")
    print(f"Wall time:         {result['wall_time_s']:.2f} s")
    print(f"Throughput:        {result['requests_per_sec']:.1f} requests/sec")
    print(f"Latency mean/p50:  {result['latency_mean_ms']:.1f} / {result['latency_p50_ms']:.1f} ms")
    print(f"Latency p95/p99:   {result['latency_p95_ms']:.1f} / {result['latency_p99_ms']:.1f} ms")
    print(f"Latency max:       {result['latency_max_ms']:.1f} ms")


def print_scaling_discussion(write_result: dict) -> None:
    measured_rps = write_result["requests_per_sec"]
    print("\n" + "=" * 70)
    print("HONEST DISCUSSION: FROM THIS MEASUREMENT TO A '100K USERS' TARGET")
    print("=" * 70)
    print(f"""
This sandbox measured ~{measured_rps:.0f} write-requests/sec from a SINGLE
uvicorn worker process writing to a SINGLE SQLite file, with per-athlete
risk-tracker state held in an in-memory Python dict (see
src/api/main.py, `_trackers`).

That architecture has three concrete, known bottlenecks, each with a
known (not speculative) fix:

1. SQLite write-serialization. SQLite allows only one writer at a time;
   every POST that writes a row queues behind the others. Fix: Postgres
   (already supported -- see src/db/models.get_engine, just needs
   DATABASE_URL set to a real Postgres instance instead of the SQLite
   default), which handles concurrent writers natively.

2. Single process. One uvicorn worker uses one CPU core and holds all
   state in its own memory. Fix: run multiple uvicorn/gunicorn workers
   behind a load balancer -- standard horizontal scaling, not a research
   problem. This requires item 3 below to be fixed first, because...

3. In-memory `_trackers` dict. Right now, the athlete's rolling risk
   history lives in the memory of whichever process handled their last
   request. Add a second worker process and an athlete's requests could
   bounce between workers with inconsistent state. Fix: move that state
   into Redis (a standard shared-cache store) or recompute it from the
   database on every request (simpler, more DB load, no shared-state bug
   class) -- a well-understood pattern, not a novel one.

None of these fixes are exotic -- they're the standard playbook for
scaling any FastAPI/Django/Rails app from "single demo instance" to
"many concurrent users," and all three are things any backend engineer
would do before a pilot goes from one hospital to many. What this load
test does NOT do is prove any specific number of users the FIXED
architecture would support -- that depends on the real database server's
hardware, the real network, and real traffic patterns, none of which
exist yet. The honest claim is: "the current single-instance demo has
these three specific, fixable bottlenecks, not that scaling is
unknown or impossible."
""")


async def main_async(args: argparse.Namespace) -> None:
    proc = start_server(args.port)
    base_url = BASE_URL.format(port=args.port)
    try:
        print(f"Server started at {base_url} (PID {proc.pid})")

        health_result = await run_load_test(base_url, args.n_requests, args.concurrency, endpoint="health")
        print_results("Read-only /health endpoint", health_result)

        write_result = await run_load_test(base_url, args.n_requests, args.concurrency, endpoint="athletes")
        print_results("Write endpoint: POST /athletes (DB insert)", write_result)

        print_scaling_discussion(write_result)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-requests", type=int, default=200, help="Total requests to send per phase")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent in-flight requests")
    parser.add_argument("--port", type=int, default=8899, help="Port to run the test server on")
    args = parser.parse_args()
    asyncio.run(main_async(args))
