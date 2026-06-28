import asyncio
import time
import httpx

API_URL = "http://localhost:8000"
NUM_REQUESTS = 10


async def make_search_request(client, index):
    """Fires a single search request."""
    payload = {"query": f"Tell me about topic {index}", "top_k": 3}

    # We use a higher timeout just in case the API provider (Gemini) throttles us slightly
    response = await client.post(f"{API_URL}/search", json=payload, timeout=30.0)

    if response.status_code != 200:
        print(f"Request {index} failed with status {response.status_code}")

    return response.status_code


async def run_sequential(client):
    print(f"--- Running {NUM_REQUESTS} Sequential Requests ---")
    start_time = time.time()

    for i in range(NUM_REQUESTS):
        await make_search_request(client, i)
        print(f"Finished request {i+1}/{NUM_REQUESTS}")

    duration = time.time() - start_time
    print(f"Total Sequential Time: {duration:.2f} seconds\n")
    return duration


async def run_concurrent(client):
    print(f"--- Running {NUM_REQUESTS} Concurrent Requests ---")
    start_time = time.time()

    # Pack all 10 network calls into an array of tasks
    tasks = [make_search_request(client, i) for i in range(NUM_REQUESTS)]

    # asyncio.gather fires them all off at the exact same time
    await asyncio.gather(*tasks)

    duration = time.time() - start_time
    print(f"Total Concurrent Time: {duration:.2f} seconds\n")
    return duration


async def main():
    # We increase max_connections so httpx doesn't artificially bottleneck our concurrent test
    limits = httpx.Limits(max_connections=50)

    async with httpx.AsyncClient(limits=limits) as client:
        seq_time = await run_sequential(client)
        conc_time = await run_concurrent(client)

        print("====== RESULTS ======")
        print(f"Sequential: {seq_time:.2f}s")
        print(f"Concurrent: {conc_time:.2f}s")

        if conc_time > 0:
            speedup = seq_time / conc_time
            print(f"Performance: Concurrency was {speedup:.2f}x faster!")


if __name__ == "__main__":
    asyncio.run(main())
