"""
Async completion endpoint for eval workloads (OpenAI / OpenRouter).

Supports concurrent requests via AsyncOpenAI + asyncio.Semaphore,
per-item checkpointing with auto-resume, and multi-message conversations.
"""

import asyncio
import argparse
import copy
import json
import os
import re
import shutil
from time import time

from openai import AsyncOpenAI
from tqdm import tqdm

from utils import (
    get_model_abbreviation,
    load_dataset_from_file,
    save_dataset,
)


def get_args():
    parser = argparse.ArgumentParser(description="Async Eval Completion Endpoint (OpenAI / OpenRouter / local vLLM)")
    parser.add_argument("--model_path", type=str, required=True, help="Model name (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--input_file", type=str, required=True, help="Input prepared file (.jsonl or .json)")
    parser.add_argument("--engine", type=str, default="openrouter_api",
                        choices=["openai", "openrouter_api", "vllm_api"], help="API engine to use.")
    parser.add_argument("--concurrency", type=int, default=50, help="Max parallel requests.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vllm_api engine (must end with /v1). Ignored for openai/openrouter.")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter API URL.")
    parser.add_argument("--openrouter_api_key", type=str, default="",
                        help="OpenRouter API Key (or set OPENROUTER_API_KEY env var).")
    parser.add_argument("--openai_api_key", type=str, default="",
                        help="OpenAI API Key (or set OPENAI_API_KEY env var).")
    parser.add_argument("--step", type=str, default="eval", help="Pipeline step tag.")
    parser.add_argument("--api_retries", type=int, default=3, help="Retry count per API request.")
    return parser.parse_args()


args = get_args()
print(f"Async Eval Completion Endpoint. Arguments: {args}")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    raise ValueError("Input file must end with prepared.json(l) for completion pipeline.")
if args.concurrency <= 0:
    raise ValueError("--concurrency must be a positive integer.")

# Resolve API credentials and base URL based on engine
if args.engine == "openrouter_api":
    _resolved_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not _resolved_api_key:
        raise ValueError(
            "OpenRouter API Key is required. "
            "Provide --openrouter_api_key or set OPENROUTER_API_KEY env var."
        )
    _resolved_base_url = args.openrouter_url.rstrip("/")
elif args.engine == "vllm_api":
    _base_url = args.base_url.rstrip("/")
    if not _base_url.endswith("/v1"):
        raise ValueError("--base_url must end with /v1 for vllm_api engine.")
    _resolved_base_url = _base_url
    _resolved_api_key = "EMPTY"
else:  # openai
    _resolved_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not _resolved_api_key:
        raise ValueError(
            "OpenAI API Key is required. "
            "Provide --openai_api_key or set OPENAI_API_KEY env var."
        )
    _resolved_base_url = "https://api.openai.com/v1"

model_abbreviation = get_model_abbreviation(args.model_path)


# --- Output path helpers ---

def get_input_base_name(input_file):
    base_name = input_file[:input_file.rfind(".")]
    if base_name.endswith("_4prepared"):
        return base_name[:-10]
    if base_name.endswith("_prepared"):
        return base_name[:-9]
    return base_name


def build_output_paths(base_name):
    saved_file = f"{base_name}_{model_abbreviation}_results.jsonl"
    checkpoint_dir = f"{base_name}_{model_abbreviation}_results_checkpoints"
    return saved_file, checkpoint_dir


# --- Checkpoint helpers ---

def checkpoint_file_path(index, checkpoint_dir):
    return os.path.join(checkpoint_dir, f"{index:08d}.json")


def save_item_checkpoint(index, item, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    cp_path = checkpoint_file_path(index, checkpoint_dir)
    temp_path = f"{cp_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, cp_path)


def load_item_checkpoints(processed_dataset, checkpoint_dir):
    completed_indices = set()
    if not os.path.isdir(checkpoint_dir):
        return completed_indices

    for file_name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(\d+)\.json", file_name)
        if match is None:
            continue

        index = int(match.group(1))
        if index < 0 or index >= len(processed_dataset):
            continue

        file_path = os.path.join(checkpoint_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                processed_dataset[index] = json.load(f)
            completed_indices.add(index)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Failed to load checkpoint {file_path}: {e}. Reprocessing index {index}.")

    return completed_indices


# --- Async API request ---

async def request_completion(messages, client):
    for attempt in range(args.api_retries):
        try:
            completion = await client.chat.completions.create(
                model=args.model_path,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
            content = completion.choices[0].message.content
            return content if content else ""
        except Exception as e:
            print(f"Request attempt {attempt + 1}/{args.api_retries} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    return ""


async def process_item(item, client):
    processed_item = copy.deepcopy(item)
    messages = processed_item.get("messages", [])
    if not messages:
        processed_item["messages"] = [{"role": "assistant", "content": ""}]
        return processed_item

    response = (await request_completion(messages, client)).strip()
    processed_item["messages"] = messages + [{"role": "assistant", "content": response}]
    return processed_item


# --- Metadata ---

def add_generation_config_to_metadata(dataset):
    config_entry = {
        "model": model_abbreviation,
        "generation_params": {
            "engine": args.engine,
            "model_path": args.model_path,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "step": args.step,
        },
        "timestamp": int(time()),
    }

    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)

    return dataset


# --- Main processing loop ---

async def generate_and_update(dataset, client, checkpoint_dir):
    processed_dataset = copy.deepcopy(dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    completed_indices = load_item_checkpoints(processed_dataset, checkpoint_dir)
    if completed_indices:
        print(f"Loaded {len(completed_indices)} completed item checkpoints from {checkpoint_dir}.")

    pending_indices = [idx for idx in range(len(processed_dataset)) if idx not in completed_indices]
    if not pending_indices:
        print("No remaining items to process.")
    else:
        print(f"Processing {len(pending_indices)} items with max concurrency {args.concurrency}.")
        semaphore = asyncio.Semaphore(args.concurrency)

        async def process_index(index):
            async with semaphore:
                try:
                    return index, await process_item(processed_dataset[index], client)
                except Exception as e:
                    print(f"Failed to process index {index}: {e}")
                    fallback = copy.deepcopy(processed_dataset[index])
                    original_messages = fallback.get("messages", [])
                    fallback["messages"] = original_messages + [{"role": "assistant", "content": ""}]
                    return index, fallback

        tasks = [asyncio.create_task(process_index(idx)) for idx in pending_indices]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating completions"):
            index, processed_item = await task
            processed_dataset[index] = processed_item
            save_item_checkpoint(index, processed_item, checkpoint_dir)

    processed_dataset = add_generation_config_to_metadata(processed_dataset)
    return processed_dataset


async def main():
    dataset = load_dataset_from_file(args.input_file)
    if not isinstance(dataset, list):
        dataset = [dataset]

    base_name = get_input_base_name(args.input_file)
    saved_file, checkpoint_dir = build_output_paths(base_name)

    print(f"Dataset rows: {len(dataset)}")
    print(f"Output file: {saved_file}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    client = AsyncOpenAI(base_url=_resolved_base_url, api_key=_resolved_api_key)
    try:
        updated_dataset = await generate_and_update(dataset, client, checkpoint_dir)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"Final dataset saved to {saved_file}. Checkpoints cleaned up.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
