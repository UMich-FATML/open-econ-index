#!/usr/bin/env python3
"""
Filter pre-assigned O*NET tasks using an LLM via OpenRouter.

For each chat (summary + 3 candidate tasks from summaries2tasks.json), asks the
LLM which tasks are genuinely represented. Outputs a new JSON with only the
validated tasks (summaries2tasks-filtered.json).

This is the second step of the chat_index pipeline:
    1. map_summaries_to_tasks.py  -> summaries2tasks.json (unfiltered top-3)
    2. filter_tasks.py            -> summaries2tasks-filtered.json (LLM-filtered)

Usage:
    python filter_tasks.py                                # full run
    python filter_tasks.py --limit 100                    # test with 100 chats
    python filter_tasks.py --model google/gemma-3-12b-it:free
    python filter_tasks.py --no_us_filter                 # process all chats
"""

import argparse
import asyncio
import json
import os
import re
from typing import Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_CONCURRENCY = 50
BATCH_SAVE_SIZE = 2000

PROMPTS = {
    "default": {
        "system": (
            "You are an expert classifier matching AI chat summaries to occupational tasks. "
            "IMPORTANT: The summaries are very brief and may omit key details. Tasks are "
            "described in formal occupational language that may sound different from casual "
            "AI chat topics, but can still be the same underlying activity. For example, "
            "'calculate equilibrium constant Kp' matches 'calculate amounts of chemicals "
            "using mathematical formulas' — the specific domain differs but the core skill "
            "is the same. Be INCLUSIVE: keep any task where the core skill or activity "
            "overlaps, even if the domain or wording differs. Only reject tasks that have "
            "NO conceivable connection to the summary."
        ),
        "user": """A user had this conversation with an AI assistant (this is only a brief summary — the actual conversation likely covered more ground):
"{summary}"

These occupational tasks were identified as potentially related. A task matches if the underlying skill or activity overlaps, even if the specific domain or terminology differs:
1. {task1}
2. {task2}
3. {task3}

Which tasks have any overlap with the conversation's activities? Return ONLY a comma-separated list of task numbers (e.g. "1,2,3") or "None" if absolutely none are related.""",
    },
}

REASONING_MODELS = {"qwen/qwen3.5-9b", "deepseek/deepseek-r1"}
DEFAULT_MAX_TOKENS = 256
REASONING_MAX_TOKENS = 4096


def parse_response(text: str, num_tasks: int = 3) -> List[int]:
    """Extract valid task indices from LLM response. Returns 0-indexed list."""
    text = text.strip().lower()
    if "none" in text and not any(c.isdigit() for c in text):
        return []
    numbers = [int(n) for n in re.findall(r"[1-3]", text)]
    valid = sorted(set(n - 1 for n in numbers if 1 <= n <= num_tasks))
    return valid


def get_prompts(model: str) -> Dict[str, str]:
    return PROMPTS.get(model, PROMPTS["default"])


async def filter_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    chat_id: str,
    summary: str,
    tasks: List[str],
    prompts: Dict[str, str],
) -> Dict:
    user_content = prompts["user"].format(
        summary=summary,
        task1=tasks[0] if len(tasks) > 0 else "(empty)",
        task2=tasks[1] if len(tasks) > 1 else "(empty)",
        task3=tasks[2] if len(tasks) > 2 else "(empty)",
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=REASONING_MAX_TOKENS if model in REASONING_MODELS else DEFAULT_MAX_TOKENS,
            )
            usage = response.usage
            content = response.choices[0].message.content or ""
            kept_indices = parse_response(content, len(tasks))
            kept_tasks = [tasks[i] for i in kept_indices]
            return {
                "id": chat_id,
                "kept_tasks": kept_tasks,
                "raw": content,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            # Fail-open: keep all tasks on error
            return {"id": chat_id, "kept_tasks": list(tasks), "error": str(e)}


async def process_batch(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    items: List[Dict],
    prompts: Dict[str, str],
    timeout_seconds: float = 45.0,
) -> List[Dict]:
    async def with_timeout(item):
        try:
            return await asyncio.wait_for(
                filter_one(
                    client, semaphore, model, item["id"], item["summary"], item["tasks"], prompts
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "id": item["id"],
                "kept_tasks": list(item["tasks"]),
                "error": "timeout",
                "input_tokens": 0,
                "output_tokens": 0,
            }

    coros = [with_timeout(it) for it in items]
    return await tqdm_asyncio.gather(*coros, desc="LLM calls")


def main():
    parser = argparse.ArgumentParser(
        description="Filter top-k task matches via OpenRouter LLM."
    )
    parser.add_argument(
        "--input_file", default="summaries2tasks.json",
        help="Input JSON from map_summaries_to_tasks.py",
    )
    parser.add_argument(
        "--output_file", default="summaries2tasks-filtered.json",
        help="Output filtered JSON",
    )
    parser.add_argument(
        "--progress_file", default="filter_progress.json",
        help="Resumable progress tracker",
    )
    parser.add_argument(
        "--us_hash_file", default="us-hash.json",
        help="JSON list of conversation_hashes to keep (US-only filter)",
    )
    parser.add_argument(
        "--no_us_filter", action="store_true",
        help="Skip US-only filtering; process all chats",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N chats (0 = all)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="OpenRouter model ID",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Max parallel API requests",
    )
    parser.add_argument(
        "--timeout", type=float, default=45.0,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var is required. Set it or load from .env."
        )

    print(f"Loading {args.input_file}...")
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)
    print(f"Total entries: {len(raw_data):,}")

    if not args.no_us_filter:
        if not os.path.exists(args.us_hash_file):
            raise FileNotFoundError(
                f"{args.us_hash_file} not found. Provide --us_hash_file or pass --no_us_filter."
            )
        print(f"Loading US hashes from {args.us_hash_file}...")
        with open(args.us_hash_file, "r") as f:
            us_hashes = set(json.load(f))
        print(f"US conversations: {len(us_hashes):,}")
        keys = [k for k in raw_data if k in us_hashes]
    else:
        keys = list(raw_data.keys())
    print(f"Chats to process: {len(keys):,}")

    already_done = set()
    if os.path.exists(args.progress_file):
        with open(args.progress_file, "r") as f:
            already_done = set(json.load(f))
        print(f"Resuming — {len(already_done):,} already processed")

    keys = [k for k in keys if k not in already_done]
    if args.limit > 0:
        keys = keys[: args.limit]
    print(f"Chats remaining: {len(keys):,}")

    if not keys:
        print("Nothing to do.")
        return

    items = []
    for k in keys:
        entry = raw_data[k]
        tasks = entry[0]
        summary = entry[1][0]
        items.append({"id": k, "summary": summary, "tasks": tasks})

    prompts = get_prompts(args.model)
    print(f"Using {'model-specific' if args.model in PROMPTS else 'default'} prompt for {args.model}")

    existing_output = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            existing_output = json.load(f)

    error_count = 0
    total_in = 0
    total_out = 0
    for batch_start in range(0, len(items), BATCH_SAVE_SIZE):
        batch = items[batch_start : batch_start + BATCH_SAVE_SIZE]
        batch_num = batch_start // BATCH_SAVE_SIZE + 1
        total_batches = (len(items) + BATCH_SAVE_SIZE - 1) // BATCH_SAVE_SIZE
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} chats) ---")

        # Fresh client per batch avoids event-loop binding issues
        client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=30.0)
        semaphore = asyncio.Semaphore(args.concurrency)

        results = asyncio.run(
            process_batch(client, semaphore, args.model, batch, prompts, args.timeout)
        )

        batch_timeouts = 0
        for r in results:
            cid = r["id"]
            original = raw_data[cid]
            total_in += r.get("input_tokens", 0)
            total_out += r.get("output_tokens", 0)
            if "error" in r:
                error_count += 1
                if r["error"] == "timeout":
                    batch_timeouts += 1
            existing_output[cid] = [r["kept_tasks"], original[1]]
            already_done.add(cid)

        with open(args.output_file, "w") as f:
            json.dump(existing_output, f)
        with open(args.progress_file, "w") as f:
            json.dump(list(already_done), f)

        print(
            f"Saved. Total processed: {len(already_done):,}, "
            f"errors: {error_count}, timeouts: {batch_timeouts}"
        )

    total = len(existing_output)
    empty = sum(1 for v in existing_output.values() if len(v[0]) == 0)
    avg_tasks = sum(len(v[0]) for v in existing_output.values()) / max(total, 1)
    print(f"\n--- Done ---")
    print(f"Total processed: {total:,}")
    print(f"Avg tasks kept: {avg_tasks:.2f} / 3")
    print(f"Chats with 0 tasks (N/A): {empty:,} ({empty / max(total, 1) * 100:.1f}%)")
    if total_in > 0:
        n = len(items)
        print(f"\nToken usage (this run, {n} chats):")
        print(f"  Avg input:  {total_in / n:.0f} tokens/chat")
        print(f"  Avg output: {total_out / n:.0f} tokens/chat")
        print(f"  Total:      {total_in:,} in / {total_out:,} out")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
