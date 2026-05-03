"""
Unified Agent Script -- Single-Turn and Multi-Turn Processors

Merges:
  - completion_openai_agent.py   (single-turn agent processor)
  - completion_multiagent.py     (multi-turn Student-User loop processor)

CLI toggles:
  --mode   single|multi   (default: multi)
  --tools  real|virtual   (default: virtual)
"""

import torch
import os
import sys
import argparse
import copy
import json
import re
import types
import asyncio
import base64
import signal
import atexit
import shutil
import threading
import traceback
import concurrent.futures
from glob import glob
from time import sleep, time
from tqdm import tqdm
from virtual_tools import VirtualToolBackend, create_dynamic_virtual_tool
from wrapt_timeout_decorator import timeout as wrapt_timeout

from utils import (
    load_dataset_from_file,
    save_dataset,
    validate_api_pool_from_file,
    get_model_abbreviation,
    safe_save_checkpoint,
)

# Suppress OpenAI Agent SDK tracing logs before importing
import logging
logging.getLogger("openai.agents").setLevel(logging.ERROR)
logging.getLogger("agents").setLevel(logging.ERROR)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# OpenAI Agent imports
from agents.mcp import MCPServerStreamableHttp
from agents.run_context import RunContextWrapper
from agents import Agent, OpenAIResponsesModel, Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
set_tracing_disabled(True)
from openai import AsyncClient, AsyncOpenAI
from typing import Dict, Any, List, Optional
from pydantic import create_model, Field, BaseModel

# Check if agents library is installed
try:
    import agents
except ImportError:
    print("agents library is not installed. Please install it.")
    exit(1)


# ================================================================
# Signal / cleanup handlers
# ================================================================

def cleanup_mcp_resources():
    """Clean up MCP resources on exit."""
    try:
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            pass
    except Exception:
        pass


def signal_handler(signum, frame):
    cleanup_mcp_resources()
    os._exit(0)


atexit.register(cleanup_mcp_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ================================================================
# Argument parsing
# ================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Unified Agent Response Generation Manager.")

    # --- NEW toggles ---
    parser.add_argument("--mode", type=str, default="multi", choices=["single", "multi"],
                        help="Processing mode: 'single' for single-turn, 'multi' for multi-turn Student-User loop (default: multi)")
    parser.add_argument("--tools", type=str, default="virtual", choices=["real", "virtual"],
                        help="Tool backend: 'real' for live MCP servers, 'virtual' for LLM-simulated tools (default: virtual)")

    # --- Shared arguments (from both scripts) ---
    parser.add_argument("--model_path", type=str, default="openai/gpt-4o",
                        help="Model path for inference")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) of rows to process.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional number of rows to process from start_idx.")
    parser.add_argument("--checkpoint_every", type=int, default=16, help="Save checkpoint every n completed items (multi mode)")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL ending with /v1.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--smithery_api_key", type=str, default="", help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json",
                        help="Path to Smithery API pool JSON file")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of parallel workers (default: use API pool size)")

    # Generation Parameters
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Use agent inference")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for each item processing (default: 90 for single, 900 for multi)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for each item processing")
    parser.add_argument("--fncall_prompt_type", type=str, default="nous", help="Function call prompt type (single mode)")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hint (single mode)")
    parser.add_argument("--enable_irrelevant_warning", action="store_true", help="Enable irrelevant warning (single mode)")
    parser.add_argument("--max_turns", type=int, default=None,
                        help="Maximum number of agent tool-call turns (default: 10 for single, 15 for multi)")

    # Tool parameters
    parser.add_argument("--virtual_tools", action="store_true",
                        help="DEPRECATED: use --tools virtual instead. Kept for backward compatibility.")
    parser.add_argument("--virtual_tool_model", type=str, default=None,
                        help="Model for virtual tool simulation (default: same as --model_path)")
    parser.add_argument("--mcp_server_dir", type=str, default="../mcp_servers/smithery_mcp_servers_0210",
                        help="Path to directory of MCP server JSON files")

    # Multi-agent specific parameters
    parser.add_argument("--user_model", type=str, default="openai/gpt-4o",
                        help="Deprecated: ignored; User agent uses --model_path")
    parser.add_argument("--user_max_turns", type=int, default=5,
                        help="Maximum number of Student-User conversation turns (multi mode)")
    parser.add_argument("--user_prompt_template", type=str, default=None,
                        help="Path to user prompt template (default: prompts/user.md relative to this script)")
    parser.add_argument("--student_prompt_template", type=str, default=None,
                        help="Path to student prompt template (default: prompts/student.md relative to this script)")

    return parser.parse_args()


args = get_args()

# ---- Apply mode-appropriate defaults for None values ----
if args.timeout is None:
    args.timeout = 90 if args.mode == "single" else 900

if args.max_turns is None:
    args.max_turns = 10 if args.mode == "single" else 15

# Reconcile --tools and legacy --virtual_tools flag
if args.virtual_tools:
    args.tools = "virtual"
# Set the canonical boolean for internal use
args.virtual_tools = (args.tools == "virtual")

args.virtual_tool_model = args.virtual_tool_model or args.model_path

if args.user_model != args.model_path:
    print(
        f"  --user_model ({args.user_model}) is deprecated and ignored. "
        f"Using --model_path ({args.model_path}) for User agent."
    )

print(f"Unified Agent Manager [mode={args.mode}, tools={args.tools}]. Arguments: {args}")

# ---- Input validation ----
if args.input_file is None:
    raise ValueError("Please specify the input file path.")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l) for completion pipeline.")
    exit(1)
if args.start_idx < 0:
    raise ValueError("--start_idx must be a non-negative integer.")
if args.batch_size is not None and args.batch_size <= 0:
    raise ValueError("--batch_size must be a positive integer.")

normalized_base_url = args.base_url.rstrip("/").removesuffix("/chat/completions")
if not normalized_base_url.endswith("/v1"):
    raise ValueError("--base_url must end with /v1.")
args.base_url = normalized_base_url

# ---- Derived constants ----
INPUT_FILE_NAME = args.input_file
CHECKPOINT_EVERY = args.checkpoint_every

model_abbreviation = get_model_abbreviation(args.model_path)
if args.mode == "multi":
    config_str = (
        f"{model_abbreviation}_multiagent_pfc" if args.parallel_function_calls
        else f"{model_abbreviation}_multiagent_sfc"
    )
else:
    config_str = (
        f"{model_abbreviation}_{args.reasoning_effort}_pfc" if args.parallel_function_calls
        else f"{model_abbreviation}_{args.reasoning_effort}_sfc"
    )

# Global API pool variable
smithery_api_pool = None

# Row-id keyed map of full question-generation conversation histories (single mode).
QUESTION_GEN_HISTORY_BY_ROW_ID = {}
QUESTION_GEN_HISTORY_SOURCE_FILE = None
QUESTION_GEN_HISTORY_READY = False


# ================================================================
# Shared: API pool management
# ================================================================

def load_and_validate_smithery_api_pool(pool_file_path):
    """Load Smithery API pool from JSON file (non-blocking)."""
    global smithery_api_pool

    print("=" * 50)
    print("SMITHERY API POOL CHECK (Non-blocking)")
    print("=" * 50)

    try:
        if not os.path.exists(pool_file_path):
            print(f"  API pool file {pool_file_path} not found.")
            print("   Proceeding without API pool (using args or virtual tools).")
            smithery_api_pool = []
            return []

        print(f"  Found {pool_file_path}. Attempting validation...")
        try:
            results = validate_api_pool_from_file(pool_file_path)

            if "error" in results:
                print(f"  API pool validation warning: {results['error']}")
                smithery_api_pool = []
                return []

            with open(pool_file_path, 'r') as f:
                original_data = json.load(f)
                original_pool = original_data.get('api_pool', [])

            valid_pool = []
            for result in results['results']:
                if result['valid']:
                    for original_entry in original_pool:
                        if original_entry['profile'] == result['profile']:
                            valid_pool.append(original_entry)
                            break

            smithery_api_pool = valid_pool
            print(f"  Loaded {len(smithery_api_pool)} valid API keys from pool.")
            return smithery_api_pool

        except Exception as e:
            print(f"  Validation check failed: {e}")
            smithery_api_pool = []
            return []

    except Exception as e:
        print(f"  Unexpected error loading pool: {e}")
        smithery_api_pool = []
        return []


def get_api_key_for_worker(worker_id):
    """Get API key and profile for a specific worker (round-robin)."""
    if smithery_api_pool and len(smithery_api_pool) > 0:
        pool_entry = smithery_api_pool[worker_id % len(smithery_api_pool)]
        return pool_entry['api_key'], pool_entry['profile']
    else:
        return args.smithery_api_key, args.smithery_profile


# ================================================================
# Shared: MCP URL construction
# ================================================================

def construct_mcp_server_url(server_info, api_key=None, profile=None):
    """Construct MCP server URL from server info."""
    if not server_info:
        return None

    server_url = server_info.get('python_sdk_url', '')
    if not server_url:
        return None

    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile

    mcp_config = server_info.get('python_sdk_config', "")
    if mcp_config == "":
        mcp_config = {"debug": False}
    else:
        try:
            mcp_config = json.loads(mcp_config)
        except json.JSONDecodeError:
            mcp_config = {"debug": False}

    config_b64 = base64.b64encode(json.dumps(mcp_config).encode()).decode()
    if "{config_b64}" in server_url:
        server_url = server_url.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in server_url:
        server_url = server_url.replace("{smithery_api_key}", api_key)
    if "{smithery_profile}" in server_url:
        server_url = server_url.replace("{smithery_profile}", profile)
    elif "&profile=" not in server_url and "profile=" not in server_url:
        server_url += f"&profile={profile}"

    return server_url


def construct_mcp_url_from_source(server_info, api_key=None, profile=None):
    """Construct MCP server URL from step 1.1 format data (source_file_path)."""
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile

    deployment_url = None
    source_path = server_info.get('source_file_path', '')
    if source_path:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.normpath(os.path.join(script_dir, source_path))
            if os.path.exists(resolved_path):
                with open(resolved_path, 'r') as f:
                    source_data = json.load(f)
                server_data = source_data.get('server', {})
                connections = server_data.get('connections', [])
                if connections:
                    deployment_url = connections[0].get('deploymentUrl', '')
                if not deployment_url:
                    dep = server_data.get('deploymentUrl', '')
                    if dep:
                        deployment_url = dep.rstrip('/') + '/mcp'
        except Exception as e:
            print(f"  Warning: Could not load source file {source_path}: {e}")

    if not deployment_url:
        return None

    config = {"debug": False}
    config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
    server_url = f"{deployment_url}?config={config_b64}&api_key={api_key}&profile={profile}"
    return server_url


# ================================================================
# Shared: Question-generation history (single-turn mode)
# ================================================================

def normalize_row_id_key(row_id):
    """Normalize row_id to a stable string key for joins."""
    if row_id is None:
        return None
    try:
        return str(int(row_id))
    except (ValueError, TypeError):
        return str(row_id)


def resolve_sanitized_file(prepared_input_file):
    """Resolve the sibling *_3sanitized.jsonl corresponding to a prepared input file."""
    output_dir = os.path.dirname(prepared_input_file) or "."
    input_basename = os.path.basename(prepared_input_file)
    stem, _ = os.path.splitext(input_basename)

    if stem.endswith("_4prepared"):
        base_stem = stem[:-10]
    elif stem.endswith("_prepared"):
        base_stem = stem[:-9]
    else:
        base_stem = stem

    exact_candidate = os.path.join(output_dir, f"{base_stem}_3sanitized.jsonl")
    if os.path.exists(exact_candidate):
        return exact_candidate

    prefixed = sorted(glob(os.path.join(output_dir, f"{base_stem}*_3sanitized.jsonl")))
    if prefixed:
        return prefixed[0]

    any_sanitized = sorted(glob(os.path.join(output_dir, "*_3sanitized.jsonl")))
    if any_sanitized:
        return any_sanitized[0]

    return None


def load_sanitized_row_id_keys(sanitized_file):
    """Load row_id keys from *_3sanitized.jsonl."""
    row_id_keys = set()
    if not sanitized_file or not os.path.exists(sanitized_file):
        return row_id_keys

    with open(sanitized_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is not None:
                row_id_keys.add(key)
    return row_id_keys


def count_results_row_id_overlap(results_file, row_id_keys):
    """Count row_id overlap between a *_results.jsonl file and sanitized row ids."""
    if not row_id_keys:
        return 0

    overlap = 0
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is not None and key in row_id_keys:
                overlap += 1
    return overlap


def select_results_file(parent_dir, sanitized_file, row_id_keys):
    """Select parent-dir *_results.jsonl that best matches sanitized row_ids."""
    candidates = sorted(glob(os.path.join(parent_dir, "*_results.jsonl")))
    if not candidates:
        return None, 0

    expected_file = None
    if sanitized_file:
        sanitized_name = os.path.basename(sanitized_file)
        if sanitized_name.endswith("_3sanitized.jsonl"):
            prefix = sanitized_name[:-len("_3sanitized.jsonl")]
            expected_path = os.path.join(parent_dir, f"{prefix}_results.jsonl")
            if os.path.exists(expected_path):
                expected_file = expected_path

    scored = []
    for path in candidates:
        overlap = count_results_row_id_overlap(path, row_id_keys)
        expected_bonus = 1 if expected_file and path == expected_file else 0
        scored.append((overlap, expected_bonus, path))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_overlap, _, best_file = scored[0]
    return best_file, best_overlap


def load_question_generation_history_by_row_id(prepared_input_file):
    """Load row_id -> full messages map from parent-dir results files."""
    output_dir = os.path.dirname(prepared_input_file) or "."
    parent_dir = os.path.dirname(output_dir) or "."

    sanitized_file = resolve_sanitized_file(prepared_input_file)
    if not sanitized_file:
        print("  Could not find *_3sanitized.jsonl near input; question-gen history lookup disabled.")
        return {}, None

    row_id_keys = load_sanitized_row_id_keys(sanitized_file)
    if not row_id_keys:
        print(f"  No row_id values found in sanitized file: {sanitized_file}")
        return {}, None

    results_file, overlap = select_results_file(parent_dir, sanitized_file, row_id_keys)
    if not results_file:
        print(f"  No parent-dir *_results.jsonl found in: {parent_dir}")
        return {}, None

    history_by_row_id = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is None or key not in row_id_keys:
                continue

            messages = item.get("messages")
            if isinstance(messages, list):
                history_by_row_id[key] = messages

    print("  Question-generation history lookup:")
    print(f"   - Sanitized file: {sanitized_file}")
    print(f"   - Parent results file: {results_file}")
    print(f"   - Sanitized row_ids: {len(row_id_keys)}")
    print(f"   - Row-id overlap: {overlap}")
    print(f"   - Loaded histories: {len(history_by_row_id)}")
    return history_by_row_id, results_file


def ensure_question_generation_history_loaded():
    """Initialize row_id->messages history cache once per run."""
    global QUESTION_GEN_HISTORY_BY_ROW_ID, QUESTION_GEN_HISTORY_SOURCE_FILE, QUESTION_GEN_HISTORY_READY
    if QUESTION_GEN_HISTORY_READY:
        return

    QUESTION_GEN_HISTORY_BY_ROW_ID, QUESTION_GEN_HISTORY_SOURCE_FILE = (
        load_question_generation_history_by_row_id(args.input_file)
    )
    QUESTION_GEN_HISTORY_READY = True


# ================================================================
# Shared: Message extraction / conversion
# ================================================================

def convert_openai_agent_result_to_messages(result, original_messages, system_prompt=None):
    """Convert OpenAI Agent result to message format compatible with Qwen Agent structure."""
    all_messages = []

    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})

    all_messages.extend(original_messages)

    if hasattr(result, 'new_items') and result.new_items:
        current_reasoning = []
        matched_call_ids = set()

        for item_flow in result.new_items:
            if item_flow.type == "reasoning_item":
                raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
                if isinstance(raw_content, list):
                    for content in raw_content:
                        if hasattr(content, 'text'):
                            current_reasoning.append(content.text)

            elif item_flow.type == "tool_call_item":
                if hasattr(item_flow, 'raw_item'):
                    tool_call = {
                        "name": getattr(item_flow.raw_item, 'name', None),
                        "arguments": getattr(item_flow.raw_item, 'arguments', None),
                        "call_id": getattr(item_flow.raw_item, 'call_id', None)
                    }

                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []

                    assistant_msg = {
                        "role": "assistant",
                        "content": "",
                        "function_call": tool_call
                    }
                    all_messages.append(assistant_msg)

            elif item_flow.type == "tool_call_output_item":
                if hasattr(item_flow, 'output'):
                    try:
                        output_data = json.loads(item_flow.output)
                        if output_data.get('type') == 'text':
                            inner_data = json.loads(output_data.get('text', '{}'))
                            tool_output = json.dumps(inner_data)
                        else:
                            tool_output = item_flow.output
                    except Exception:
                        tool_output = item_flow.output

                    tool_name = 'unknown'
                    matched_call_id = None
                    if hasattr(item_flow, 'raw_item'):
                        raw = item_flow.raw_item
                        call_id = None
                        for attr in ['tool_call_id', 'call_id', 'id', 'toolCallId']:
                            if hasattr(raw, attr):
                                call_id = getattr(raw, attr)
                                break
                        if call_id is not None:
                            for prev_msg in reversed(all_messages):
                                if (prev_msg.get('role') == 'assistant' and
                                        'function_call' in prev_msg and
                                        prev_msg['function_call'].get('call_id') == call_id):
                                    tool_name = prev_msg['function_call'].get('name', 'unknown')
                                    matched_call_id = call_id
                                    break

                    if tool_name == 'unknown':
                        for prev_msg in all_messages:
                            if prev_msg.get('role') == 'assistant' and 'function_call' in prev_msg:
                                fc = prev_msg['function_call']
                                fc_call_id = fc.get('call_id')
                                if fc_call_id not in matched_call_ids:
                                    name_candidate = fc.get('name')
                                    if name_candidate:
                                        tool_name = name_candidate
                                        matched_call_id = fc_call_id
                                        break

                    if matched_call_id is not None:
                        matched_call_ids.add(matched_call_id)

                    all_messages.append({
                        "role": "function",
                        "content": tool_output,
                        "name": tool_name
                    })

            elif item_flow.type == "message_output_item":
                raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
                if isinstance(raw_content, list):
                    message_texts = []
                    for content in raw_content:
                        if hasattr(content, 'text'):
                            message_texts.append(content.text)

                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []

                    final_content = "\n".join(message_texts)
                    if final_content.strip():
                        final_msg = {"role": "assistant", "content": final_content}
                        all_messages.append(final_msg)

    # Fallback if no assistant messages were produced
    new_messages_start = len(original_messages) + (1 if system_prompt else 0)
    if not any(msg.get('role') == 'assistant' and msg.get('content') for msg in all_messages[new_messages_start:]):
        final_msg = {"role": "assistant", "content": result.final_output}

        reasoning_content = []
        if hasattr(result, 'new_items') and result.new_items:
            for item_flow in result.new_items:
                if item_flow.type == "reasoning_item":
                    raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
                    if isinstance(raw_content, list):
                        for content in raw_content:
                            if hasattr(content, 'text'):
                                reasoning_content.append(content.text)

        if reasoning_content:
            all_messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": "\n".join(reasoning_content)
            })

        all_messages.append(final_msg)

    return all_messages


def extract_new_messages_from_result(result):
    """Extract only the NEW messages from an agent result (multi-turn mode)."""
    new_messages = []

    if not hasattr(result, 'new_items') or not result.new_items:
        if result.final_output:
            new_messages.append({"role": "assistant", "content": result.final_output})
        return new_messages

    current_reasoning = []
    matched_call_ids = set()

    for item_flow in result.new_items:
        if item_flow.type == "reasoning_item":
            raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
            if isinstance(raw_content, list):
                for content in raw_content:
                    if hasattr(content, 'text'):
                        current_reasoning.append(content.text)

        elif item_flow.type == "tool_call_item":
            if hasattr(item_flow, 'raw_item'):
                tool_call = {
                    "name": getattr(item_flow.raw_item, 'name', None),
                    "arguments": getattr(item_flow.raw_item, 'arguments', None),
                    "call_id": getattr(item_flow.raw_item, 'call_id', None)
                }
                if current_reasoning:
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "\n".join(current_reasoning)
                    })
                    current_reasoning = []
                new_messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": tool_call
                })

        elif item_flow.type == "tool_call_output_item":
            if hasattr(item_flow, 'output'):
                try:
                    output_data = json.loads(item_flow.output)
                    if output_data.get('type') == 'text':
                        inner_data = json.loads(output_data.get('text', '{}'))
                        tool_output = json.dumps(inner_data)
                    else:
                        tool_output = item_flow.output
                except Exception:
                    tool_output = item_flow.output

                tool_name = 'unknown'
                matched_call_id = None
                if hasattr(item_flow, 'raw_item'):
                    raw = item_flow.raw_item
                    call_id = None
                    for attr in ['tool_call_id', 'call_id', 'id', 'toolCallId']:
                        if hasattr(raw, attr):
                            call_id = getattr(raw, attr)
                            break
                    if call_id is not None:
                        for prev_msg in reversed(new_messages):
                            if (prev_msg.get('role') == 'assistant' and
                                    'function_call' in prev_msg and
                                    prev_msg['function_call'].get('call_id') == call_id):
                                tool_name = prev_msg['function_call'].get('name', 'unknown')
                                matched_call_id = call_id
                                break

                if tool_name == 'unknown':
                    for prev_msg in new_messages:
                        if prev_msg.get('role') == 'assistant' and 'function_call' in prev_msg:
                            fc = prev_msg['function_call']
                            fc_call_id = fc.get('call_id')
                            if fc_call_id not in matched_call_ids:
                                name_candidate = fc.get('name')
                                if name_candidate:
                                    tool_name = name_candidate
                                    matched_call_id = fc_call_id
                                    break

                if matched_call_id is not None:
                    matched_call_ids.add(matched_call_id)

                new_messages.append({
                    "role": "function",
                    "content": tool_output,
                    "name": tool_name
                })

        elif item_flow.type == "message_output_item":
            raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
            if isinstance(raw_content, list):
                message_texts = []
                for content in raw_content:
                    if hasattr(content, 'text'):
                        message_texts.append(content.text)

                if current_reasoning:
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "\n".join(current_reasoning)
                    })
                    current_reasoning = []

                final_content = "\n".join(message_texts)
                if final_content.strip():
                    new_messages.append({"role": "assistant", "content": final_content})

    # Flush remaining reasoning
    if current_reasoning:
        new_messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": "\n".join(current_reasoning)
        })

    # Fallback
    if not any(msg.get('role') == 'assistant' and msg.get('content') for msg in new_messages):
        if result.final_output:
            new_messages.append({"role": "assistant", "content": result.final_output})

    return new_messages


# ================================================================
# Shared: Qwen-compatible system prompt
# ================================================================

def qwen_compatible_system_prompt_generator(tools):
    """Generate a Qwen-compatible system prompt from tool specs."""
    function_schemas = []
    for tool in tools or []:
        name = getattr(tool, 'name', None) or ''
        description = getattr(tool, 'description', None) or ''
        params_schema = getattr(tool, 'params_json_schema', None) or {"type": "object", "properties": {}}
        function_schemas.append({"name": name, "description": description, "parameters": params_schema})

    tool_descs_wrapped = [{"type": "function", "function": fs} for fs in function_schemas]
    tool_descs_str = "\n".join(json.dumps(d, ensure_ascii=False) for d in tool_descs_wrapped)

    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tool_descs_str}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>"
    )


# ================================================================
# Shared: Output path / dataset utilities
# ================================================================

def sort_dataset_by_row_id(dataset):
    """Sort dataset by row_id from metadata."""
    def get_sort_key(item):
        metadata = item.get('metadata', {})
        row_id = metadata.get('row_id')
        if row_id is not None:
            try:
                return int(row_id)
            except (ValueError, TypeError):
                return float('inf'), str(row_id)
        return float('inf'), ''
    return sorted(dataset, key=get_sort_key)


def get_input_base_name(input_file):
    base_name = input_file[: input_file.rfind(".")]
    if base_name.endswith("_4prepared"):
        return base_name[:-10]
    if base_name.endswith("_prepared"):
        return base_name[:-9]
    return base_name


def resolve_processing_range(total_rows):
    if total_rows <= 0:
        raise ValueError(f"Input dataset is empty: {args.input_file}")
    if args.start_idx >= total_rows:
        raise ValueError(
            f"--start_idx ({args.start_idx}) must be smaller than dataset size ({total_rows})."
        )
    start_idx = args.start_idx
    requested_end_idx = total_rows if args.batch_size is None else args.start_idx + args.batch_size
    end_idx = min(requested_end_idx, total_rows)
    return start_idx, requested_end_idx, end_idx


def build_output_paths(base_name, start_idx, end_idx, trial_idx=None):
    trial_suffix = f"{trial_idx}" if trial_idx is not None else ""
    range_mode = args.batch_size is not None or args.start_idx != 0
    if args.mode == "single":
        # Single mode uses a checkpoint directory
        if range_mode:
            saved_file = f"{base_name}_{config_str}_results{trial_suffix}_{start_idx}_{end_idx}.jsonl"
            checkpoint_path = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints_{start_idx}_{end_idx}"
        else:
            saved_file = f"{base_name}_{config_str}_results{trial_suffix}.jsonl"
            checkpoint_path = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints"
    else:
        # Multi mode uses a single checkpoint file
        if range_mode:
            saved_file = f"{base_name}_{config_str}_results{trial_suffix}_{start_idx}_{end_idx}.jsonl"
            checkpoint_path = f"{base_name}_{config_str}_results{trial_suffix}_{start_idx}_{end_idx}_checkpoint.json"
        else:
            saved_file = f"{base_name}_{config_str}_results{trial_suffix}.jsonl"
            checkpoint_path = f"{base_name}_{config_str}_results{trial_suffix}_checkpoint.json"
    return saved_file, checkpoint_path


def add_generation_config_to_metadata(dataset, model_short_name, generation_params):
    """Add synthetic data generation config to each item's metadata."""
    config_entry = {
        "model": model_short_name,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    return dataset


def build_generation_params(max_workers):
    """Build a generation parameters dict for metadata tagging."""
    params = {
        "base_url": args.base_url,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step,
        "agent": args.agent,
        "timeout": args.timeout,
        "max_workers": max_workers,
        "parallel_function_calls": args.parallel_function_calls,
        "reasoning_effort": args.reasoning_effort,
        "mode": args.mode,
        "tools": args.tools,
    }
    if args.mode == "multi":
        params["student_model"] = args.model_path
        params["user_model"] = args.model_path
        params["virtual_tool_model"] = args.virtual_tool_model
        params["user_max_turns"] = args.user_max_turns
        params["virtual_tools"] = args.virtual_tools
        params["start_idx"] = args.start_idx
        params["batch_size"] = args.batch_size
    return params


# ================================================================
# Shared: Checkpoint helpers (single-turn uses per-item files,
#         multi-turn uses a single JSON checkpoint file)
# ================================================================

# -- Single-turn checkpoint helpers --

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


# -- Multi-turn checkpoint helpers --

def get_checkpoint_identity(item):
    """Build a stable identity key for checkpoint dedupe/resume matching."""
    if not isinstance(item, dict):
        return None
    metadata = item.get('metadata', {})
    row_id = metadata.get('row_id')
    prompt_id = metadata.get('prompt_id')
    if row_id is not None:
        return ("row_id", row_id)
    if prompt_id:
        return ("prompt_id", prompt_id)
    messages = item.get('messages', [])
    if messages:
        user_msg = next((msg.get('content', '') for msg in messages if msg.get('role') == 'user'), '')
        if user_msg:
            return ("user_hash", hash(user_msg))
    return None


def dedupe_checkpoint_items(items):
    """Deduplicate checkpoint entries while keeping the latest occurrence."""
    if not isinstance(items, list):
        items = [items]
    deduped_by_key = {}
    passthrough = []
    for item in items:
        key = get_checkpoint_identity(item)
        if key is None:
            passthrough.append(item)
        else:
            deduped_by_key[key] = item
    deduped = list(deduped_by_key.values()) + passthrough
    return sort_dataset_by_row_id(deduped)


def build_error_item(item, error_text):
    fallback_item = copy.deepcopy(item)
    original_messages = fallback_item.get("messages", [])
    fallback_item["messages"] = original_messages + [
        {"role": "assistant", "content": f"[ERROR: {error_text}]"}
    ]
    return fallback_item


# ================================================================
# Shared: Client factory
# ================================================================

def _make_client():
    """Create an AsyncClient for the configured endpoint."""
    return AsyncClient(base_url=args.base_url, api_key=args.api_key)


# ================================================================
# Single-turn: Agent creation
# ================================================================

def create_agent_for_item(item, api_key=None, profile=None):
    """Create an OpenAI Agent for an item (single-turn mode).
    Supports both REAL MCP servers and VIRTUAL (LLM-generated) tools."""
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])

    if not mcp_servers or not isinstance(mcp_servers, list):
        return None

    client = AsyncClient(base_url=args.base_url, api_key=args.api_key)
    model = OpenAIResponsesModel(args.model_path, openai_client=client)

    # --- VIRTUAL TOOLS ---
    if args.virtual_tools:
        ensure_question_generation_history_loaded()

        print(f"  Configuring Agent with VIRTUAL tools (Agent: {args.model_path}, VirtualTool: {args.virtual_tool_model})...")
        virtual_backend = VirtualToolBackend(client, model_path=args.virtual_tool_model)
        virtual_tool_funcs = []
        registered_tool_names = set()

        question = metadata.get('question', '')
        tool_analysis = metadata.get('tool_analysis', '')
        workflow_analysis = metadata.get('cross_tool_workflow', '')

        target_tools = metadata.get('target_tools', [])
        expected_outputs_by_tool = {}
        required_tool_names = set()
        for tt in (target_tools or []):
            if isinstance(tt, dict):
                tool_key = tt.get('tool', '')
                if tool_key:
                    normalized_tool_key = tool_key.split("::", 1)[-1]
                    expected_outputs_by_tool[normalized_tool_key] = tt.get('output', '')
                    required_tool_names.add(normalized_tool_key)

        virtual_tool_diagnostics = {
            "total_tools_discovered": 0,
            "tools_registered": 0,
            "required_tools_expected": sorted(required_tool_names),
            "required_tool_errors": [],
            "optional_tool_errors": [],
        }

        row_id_key = normalize_row_id_key(metadata.get('row_id'))
        matched_history = QUESTION_GEN_HISTORY_BY_ROW_ID.get(row_id_key) if row_id_key is not None else None
        if not isinstance(matched_history, list) or not matched_history:
            matched_history = item.get("messages", [])
        if not isinstance(matched_history, list) or not matched_history:
            matched_history = [{"role": "user", "content": question}] if question else []
        tool_simulation_messages = []

        for server_info in mcp_servers:
            server_id = server_info.get('server_id', '')
            server_name = server_info.get('server_name', '')
            server_analysis = server_info.get('server_description', '')
            tools_list = server_info.get('tools', [])

            if args.mcp_server_dir and server_id:
                smithery_path = os.path.join(args.mcp_server_dir, f"{server_id}.json")
                if os.path.exists(smithery_path):
                    with open(smithery_path) as sf:
                        smithery_data = json.load(sf)
                    smithery_tools = smithery_data.get('server', {}).get('tools', [])
                    tools_list = smithery_tools if smithery_tools else tools_list
                    server_analysis = smithery_data.get('analysis', server_analysis)
                    server_name = smithery_data.get('server', {}).get('displayName', server_name)

            for tool_def in tools_list:
                virtual_tool_diagnostics["total_tools_discovered"] += 1
                if server_analysis and server_name and tool_def.get('description'):
                    tool_def = dict(tool_def)
                    tool_def['description'] = (
                        f"This tool comes from the MCP server: {server_name}.\n\n"
                        f"An analysis of this server is as follows: {server_analysis}.\n\n"
                        f"This tool has the following functionality within the MCP server: {tool_def['description']}"
                    )
                tool_raw_name = tool_def.get('name', '')
                expected_output = expected_outputs_by_tool.get(tool_raw_name, '')
                scenario_context = {
                    "conversation_history": matched_history,
                    "tool_simulation_messages": tool_simulation_messages,
                    "question": question,
                    "tool_analysis": tool_analysis,
                    "workflow_analysis": workflow_analysis,
                    "expected_output": expected_output,
                    "server_id": server_id,
                    "server_name": server_name,
                    "server_description": server_analysis,
                }
                is_required = tool_raw_name in required_tool_names
                try:
                    v_tool = create_dynamic_virtual_tool(
                        tool_def, virtual_backend, scenario_context=scenario_context,
                    )
                    virtual_tool_funcs.append(v_tool)
                    registered_tool_names.add(tool_raw_name)
                except Exception as tool_error:
                    err_entry = {
                        "tool": tool_raw_name,
                        "server_id": server_id,
                        "server_name": server_name,
                        "error": str(tool_error),
                    }
                    if is_required:
                        virtual_tool_diagnostics["required_tool_errors"].append(err_entry)
                    else:
                        virtual_tool_diagnostics["optional_tool_errors"].append(err_entry)
                    print(
                        f"  Skipping virtual tool '{tool_raw_name}' "
                        f"(required={is_required}) due to registration error: {tool_error}"
                    )

        missing_required_tools = sorted(required_tool_names - registered_tool_names)
        virtual_tool_diagnostics["tools_registered"] = len(virtual_tool_funcs)
        virtual_tool_diagnostics["missing_required_tools"] = missing_required_tools
        virtual_tool_diagnostics["tools_skipped_required"] = len(virtual_tool_diagnostics["required_tool_errors"])
        virtual_tool_diagnostics["tools_skipped_optional"] = len(virtual_tool_diagnostics["optional_tool_errors"])
        metadata["virtual_tool_diagnostics"] = virtual_tool_diagnostics

        if missing_required_tools:
            raise ValueError(
                "Required virtual tool(s) failed to register: "
                + ", ".join(missing_required_tools)
            )

        if not virtual_tool_funcs:
            print("  No tool definitions found in metadata for virtual generation.")
            return None

        return {
            "name": "OSS-Virtual-Assistant",
            "instructions": "You are a helpful assistant. Use the provided tools to answer the user query.",
            "model": model,
            "tools": virtual_tool_funcs,
            "mcp_servers_list": [],
            "virtual_tool_diagnostics": virtual_tool_diagnostics,
        }

    # --- REAL MCP SERVERS ---
    else:
        mcp_servers_list = []
        for server_info in mcp_servers:
            server_url = None
            server_details = server_info.get('server_info', {})
            if server_details:
                server_url = construct_mcp_server_url(server_details, api_key, profile)
            if not server_url:
                server_url = construct_mcp_url_from_source(server_info, api_key, profile)

            if server_url:
                safe_name = server_info.get('server_name', 'unknown').replace(' ', '-').lower()
                mcp_servers_list.append({
                    "name": safe_name,
                    "url": server_url,
                    "timeout": 600.0,
                    "sse_read_timeout": 600.0,
                    "terminate_on_close": False
                })

        if not mcp_servers_list:
            return None

        return {
            "name": "OSS-Assistant",
            "instructions": "You are a helpful assistant. Use the available tools.",
            "model": model,
            "mcp_servers_list": mcp_servers_list
        }


# ================================================================
# Multi-turn: Prompt utilities
# ================================================================

def load_user_prompt_template():
    """Load the user.md prompt template."""
    if args.user_prompt_template:
        path = args.user_prompt_template
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, 'prompts', 'user.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_student_prompt_template():
    """Load the student.md prompt template."""
    if args.student_prompt_template:
        path = args.student_prompt_template
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, 'prompts', 'student.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def format_target_tool_outputs(target_tools):
    """Format target_tools list into readable ground truth for the user agent."""
    if not target_tools or not isinstance(target_tools, list):
        return "No ground truth tool outputs available."
    parts = []
    for i, tool_entry in enumerate(target_tools, 1):
        if isinstance(tool_entry, dict):
            server = tool_entry.get('server', '')
            tool = tool_entry.get('tool', '')
            output = tool_entry.get('output', '')
            parts.append(f"Tool {i}: {server}::{tool}\nOutput: {output}")
    return "\n\n".join(parts) if parts else "No ground truth tool outputs available."


def format_withheld_info(withheld_info):
    """Format withheld_info list into a section for the user prompt template."""
    if not withheld_info or not isinstance(withheld_info, list):
        return ""
    lines = []
    for item in withheld_info:
        if isinstance(item, dict):
            param = item.get('parameter', '')
            desc = item.get('description', '')
            value = item.get('value', '')
            lines.append(f'- **{param}** ({desc}): When asked, answer: "{value}"')
    if not lines:
        return ""
    section = (
        "\n<withheld_information>\n"
        "The following information was NOT included in the initial query. "
        "Provide it naturally as the user when the assistant asks for it:\n"
        + "\n".join(lines)
        + "\n</withheld_information>\n"
    )
    return section


def format_tool_descriptions(metadata):
    """Build a concise tool-description block from mcp_servers metadata."""
    mcp_servers = metadata.get('mcp_servers', [])
    lines = []
    for srv in mcp_servers:
        srv_name = srv.get('server_name', 'Unknown')
        for tool in srv.get('tools', []):
            name = tool.get('name', '?')
            desc = tool.get('description', '') or '(no description)'
            lines.append(f"- **{srv_name}::{name}**: {desc}")
    return "\n".join(lines) if lines else "No tool descriptions available."


def build_user_agent_instructions(metadata):
    """Build user agent instructions from item metadata."""
    question = metadata.get('question', '')
    target_tools = metadata.get('target_tools', [])
    tool_analysis = metadata.get('tool_analysis', '')
    workflow = metadata.get('cross_tool_workflow', '')
    withheld_info = metadata.get('withheld_info', [])

    withheld_str = format_withheld_info(withheld_info)
    tool_desc_str = format_tool_descriptions(metadata)

    instructions = (
        USER_PROMPT_TEMPLATE
        .replace("{QUESTION}", question)
        .replace("{TOOL_ANALYSIS}", tool_analysis)
        .replace("{WORKFLOW_ANALYSIS}", workflow)
        .replace("{WITHHELD_INFO}", withheld_str)
        .replace("{TOOL_DESCRIPTIONS}", tool_desc_str)
    )
    return instructions


# ================================================================
# Multi-turn: Student agent creation
# ================================================================

def create_student_agent_config(item, client, api_key=None, profile=None):
    """Create configuration for the Student (tool-using) agent."""
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])

    if not mcp_servers or not isinstance(mcp_servers, list):
        return None

    model = OpenAIResponsesModel(args.model_path, openai_client=client)

    # --- VIRTUAL TOOLS ---
    if args.virtual_tools:
        print(f"  Configuring Student with VIRTUAL tools (Agent/User: {args.model_path}, VirtualTool: {args.virtual_tool_model})...")
        virtual_backend = VirtualToolBackend(client, model_path=args.virtual_tool_model)
        virtual_tool_funcs = []
        registered_tool_names = set()
        tool_simulation_messages = []

        question = metadata.get('question', '')
        tool_analysis = metadata.get('tool_analysis', '')
        workflow_analysis = metadata.get('cross_tool_workflow', '')
        target_tools = metadata.get('target_tools', [])
        expected_outputs_by_tool = {}
        required_tool_names = set()
        for tt in (target_tools or []):
            if isinstance(tt, dict):
                tool_key = tt.get('tool', '')
                if tool_key:
                    normalized_tool_key = tool_key.split("::", 1)[-1]
                    expected_outputs_by_tool[normalized_tool_key] = tt.get('output', '')
                    required_tool_names.add(normalized_tool_key)

        virtual_tool_diagnostics = {
            "total_tools_discovered": 0,
            "tools_registered": 0,
            "required_tools_expected": sorted(required_tool_names),
            "required_tool_errors": [],
            "optional_tool_errors": [],
        }

        for server_info in mcp_servers:
            server_id = server_info.get('server_id', '')
            server_name = server_info.get('server_name', '')
            server_analysis = server_info.get('server_description', '')
            tools_list = server_info.get('tools', [])

            if args.mcp_server_dir and server_id:
                smithery_path = os.path.join(args.mcp_server_dir, f"{server_id}.json")
                if os.path.exists(smithery_path):
                    with open(smithery_path) as sf:
                        smithery_data = json.load(sf)
                    smithery_tools = smithery_data.get('server', {}).get('tools', [])
                    tools_list = smithery_tools if smithery_tools else tools_list
                    server_analysis = smithery_data.get('analysis', server_analysis)
                    server_name = smithery_data.get('server', {}).get('displayName', server_name)

            for tool_def in tools_list:
                virtual_tool_diagnostics["total_tools_discovered"] += 1
                if server_analysis and server_name and tool_def.get('description'):
                    tool_def = dict(tool_def)
                    tool_def['description'] = (
                        f"This tool comes from the MCP server: {server_name}.\n\n"
                        f"An analysis of this server is as follows: {server_analysis}.\n\n"
                        f"This tool has the following functionality within the MCP server: {tool_def['description']}"
                    )
                tool_raw_name = tool_def.get('name', '')
                expected_output = expected_outputs_by_tool.get(tool_raw_name, '')
                scenario_ctx = {
                    'question': question,
                    'tool_analysis': tool_analysis,
                    'workflow_analysis': workflow_analysis,
                    'expected_output': expected_output,
                    'tool_simulation_messages': tool_simulation_messages,
                }
                is_required = tool_raw_name in required_tool_names
                try:
                    v_tool = create_dynamic_virtual_tool(
                        tool_def, virtual_backend, scenario_context=scenario_ctx,
                    )
                    virtual_tool_funcs.append(v_tool)
                    registered_tool_names.add(tool_raw_name)
                except Exception as tool_error:
                    err_entry = {
                        "tool": tool_raw_name,
                        "server_id": server_id,
                        "server_name": server_name,
                        "error": str(tool_error),
                    }
                    if is_required:
                        virtual_tool_diagnostics["required_tool_errors"].append(err_entry)
                    else:
                        virtual_tool_diagnostics["optional_tool_errors"].append(err_entry)
                    print(
                        f"  Skipping virtual tool '{tool_raw_name}' "
                        f"(required={is_required}) due to registration error: {tool_error}"
                    )

        missing_required_tools = sorted(required_tool_names - registered_tool_names)
        virtual_tool_diagnostics["tools_registered"] = len(virtual_tool_funcs)
        virtual_tool_diagnostics["missing_required_tools"] = missing_required_tools
        virtual_tool_diagnostics["tools_skipped_required"] = len(virtual_tool_diagnostics["required_tool_errors"])
        virtual_tool_diagnostics["tools_skipped_optional"] = len(virtual_tool_diagnostics["optional_tool_errors"])
        metadata["virtual_tool_diagnostics"] = virtual_tool_diagnostics

        if missing_required_tools:
            raise ValueError(
                "Required virtual tool(s) failed to register: "
                + ", ".join(missing_required_tools)
            )

        if not virtual_tool_funcs:
            print("  No tool definitions found for virtual generation.")
            return None

        return {
            "name": "Student-Virtual-Assistant",
            "instructions": STUDENT_PROMPT_TEMPLATE,
            "model": model,
            "tools": virtual_tool_funcs,
            "mcp_servers_list": [],
            "virtual_tool_diagnostics": virtual_tool_diagnostics,
        }

    # --- REAL MCP SERVERS ---
    else:
        mcp_servers_list = []
        for server_info in mcp_servers:
            server_url = None
            server_details = server_info.get('server_info', {})
            if server_details:
                server_url = construct_mcp_server_url(server_details, api_key, profile)
            if not server_url:
                server_url = construct_mcp_url_from_source(server_info, api_key, profile)
            if server_url:
                safe_name = server_info.get('server_name', 'unknown').replace(' ', '-').lower()
                mcp_servers_list.append({
                    "name": safe_name,
                    "url": server_url,
                    "timeout": 600.0,
                    "sse_read_timeout": 600.0,
                    "terminate_on_close": False
                })

        if not mcp_servers_list:
            return None

        return {
            "name": "Student-Assistant",
            "instructions": STUDENT_PROMPT_TEMPLATE,
            "model": model,
            "mcp_servers_list": mcp_servers_list
        }


# ================================================================
# Multi-turn: Student-User loop
# ================================================================

async def run_student_user_loop_async(
    student_agent_config,
    user_instructions,
    question,
    user_client,
    session_prefix,
    mcp_server_contexts=None,
):
    """
    Run the alternating Student-User multi-turn loop.
    Returns conversation_messages (full trajectory as list of message dicts).
    """
    # --- Create Student agent ---
    agent_kwargs = {
        "name": student_agent_config["name"],
        "instructions": student_agent_config["instructions"],
        "model": student_agent_config["model"],
    }
    if student_agent_config.get("tools"):
        agent_kwargs["tools"] = student_agent_config["tools"]
    if mcp_server_contexts:
        agent_kwargs["mcp_servers"] = mcp_server_contexts

    student_agent = Agent(**agent_kwargs)
    run_context = RunContextWrapper(context=None)

    # --- Create User agent (no tools) ---
    user_model = OpenAIResponsesModel(args.model_path, openai_client=user_client)
    user_agent = Agent(
        name="User",
        instructions=user_instructions,
        model=user_model,
    )

    # --- Initialize sessions ---
    student_session = SQLiteSession(f"student_{session_prefix}")
    await student_session.clear_session()

    user_session = SQLiteSession(f"user_{session_prefix}")
    await user_session.clear_session()

    # Pre-fill User session: User "already said" the initial question
    await user_session.add_items([{"role": "assistant", "content": question}])

    # --- Build conversation trajectory ---
    conversation_messages = [{"role": "user", "content": question}]
    system_prompt = None
    user_feedback = None

    for turn in range(args.user_max_turns):
        print(f"    Turn {turn + 1}/{args.user_max_turns}")

        # --- STUDENT TURN ---
        student_input = question if turn == 0 else user_feedback

        student_result = await Runner.run(
            student_agent,
            input=student_input,
            session=student_session,
            max_turns=args.max_turns,
        )

        # Generate system prompt once from first turn
        if system_prompt is None:
            available_tools = await student_agent.get_all_tools(run_context)
            system_prompt = qwen_compatible_system_prompt_generator(available_tools)

        # Extract new messages from this student turn
        new_msgs = extract_new_messages_from_result(student_result)
        conversation_messages.extend(new_msgs)

        student_reply = student_result.final_output or ""
        print(f"    [Student]: {student_reply[:120]}...")

        # --- USER TURN ---
        user_input = f"The assistant responded: {student_reply}"

        user_result = await Runner.run(
            user_agent,
            input=user_input,
            session=user_session,
        )
        user_feedback = user_result.final_output or ""
        print(f"    [User]: {user_feedback[:120]}...")

        # Add user message to conversation trajectory
        conversation_messages.append({"role": "user", "content": user_feedback})

        # Check for termination signal
        if "<end_conversation>" in user_feedback.lower():
            print("    Conversation ended by User agent.")
            break

    # Prepend system prompt
    if system_prompt:
        conversation_messages = [{"role": "system", "content": system_prompt}] + conversation_messages

    return conversation_messages


# ================================================================
# Single-turn: Item processing
# ================================================================

async def process_single_item_agent_async(item, api_key=None, profile=None):
    """Process a single item using agent inference (async, single-turn mode)."""
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')

    if args.enable_tool_hint:
        if "metadata" in item and "target_tools" in item["metadata"]:
            target_tools = item["metadata"].get('target_tools', "")
        else:
            target_tools = item.get("target_tools", "")
        tool_list = [tool.strip() for tool in target_tools.split(',')]
        tool_list = [tool.split('::')[1] if '::' in tool else tool for tool in tool_list]
        tool_list = [f"{tool}" for tool in tool_list]
        tool_list = ", ".join(tool_list)
        print(f"  Tool list: {tool_list}")

    message = item["messages"]
    if message[0]['role'] == 'system':
        message = message[1:]

    user_messages = [msg for msg in message if msg.get('role') == 'user']
    if user_messages:
        user_content = user_messages[-1]['content']
    else:
        raise ValueError("No user messages found")

    agent_config = None
    if args.agent:
        agent_config = create_agent_for_item(item, api_key, profile)
    if agent_config and agent_config.get("virtual_tool_diagnostics"):
        item.setdefault("metadata", {})["virtual_tool_diagnostics"] = agent_config["virtual_tool_diagnostics"]

    if agent_config:
        try:
            print(f"  Running OpenAI agent inference for item {prompt_id}...")

            if args.enable_tool_hint:
                if tool_list:
                    tool_hint = f'\n\nWe need to use the following tools: {tool_list}.'
                else:
                    tool_hint = '\n\nWe need to use the provided tools.'
                user_content = user_content + tool_hint

            if args.enable_irrelevant_warning:
                user_content = user_content + '\n\nUse tools only if they are relevant. Otherwise, do not use them.'

            server_configs = agent_config["mcp_servers_list"]
            mcp_servers = []
            server_contexts = []

            async def create_mcp_servers():
                mcp_servers = []
                server_contexts = []
                failed_servers = []

                for server_config in server_configs:
                    try:
                        mcp_server_context = MCPServerStreamableHttp(
                            name=server_config["name"],
                            params={
                                "url": server_config["url"],
                                "headers": {
                                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                                },
                                "timeout": server_config.get("timeout", 600.0),
                                "sse_read_timeout": server_config.get("sse_read_timeout", 600.0),
                                "terminate_on_close": server_config.get("terminate_on_close", False)
                            },
                            client_session_timeout_seconds=args.timeout,
                        )
                        mcp_server = await mcp_server_context.__aenter__()
                        mcp_servers.append(mcp_server)
                        server_contexts.append(mcp_server_context)
                    except Exception as conn_error:
                        failed_servers.append(server_config["name"])
                        print(f"  Skipping MCP server '{server_config['name']}': {conn_error}")

                if failed_servers:
                    print(f"   Failed servers: {', '.join(failed_servers)} ({len(mcp_servers)}/{len(server_configs)} connected)")

                return mcp_servers, server_contexts

            try:
                if server_configs:
                    mcp_servers, server_contexts = await create_mcp_servers()

                if not mcp_servers and not agent_config.get("tools"):
                    raise Exception(f"All {len(server_configs)} MCP server(s) failed to connect -- cannot proceed without real tools")

                try:
                    agent_kwargs = {
                        "name": agent_config["name"],
                        "instructions": agent_config["instructions"],
                        "model": agent_config["model"],
                    }

                    if mcp_servers:
                        agent_kwargs["mcp_servers"] = mcp_servers
                    if agent_config.get("tools"):
                        agent_kwargs["tools"] = agent_config["tools"]

                    agent = Agent(**agent_kwargs)
                    run_context = RunContextWrapper(context=None)

                    if len(message) > 1:
                        session = SQLiteSession(f"conversation_{prompt_id}")
                        await session.clear_session()

                        history_items = []
                        for msg in message[:-1]:
                            if msg['role'] == 'user':
                                history_items.append({"role": "user", "content": msg['content']})
                            elif msg['role'] == 'assistant':
                                history_items.append({"role": "assistant", "content": msg['content']})
                            elif msg['role'] == 'function':
                                function_name = msg.get('name', 'unknown_function')
                                history_items.append({
                                    "role": "assistant",
                                    "content": f"[Function {function_name} returned: {msg['content']}]"
                                })

                        if history_items:
                            await session.add_items(history_items)

                        result = await Runner.run(agent, input=user_content, session=session, max_turns=args.max_turns)
                    else:
                        result = await Runner.run(agent, input=user_content, max_turns=args.max_turns)

                    available_tools = await agent.get_all_tools(run_context)
                    system_prompt = qwen_compatible_system_prompt_generator(available_tools)

                    all_messages = convert_openai_agent_result_to_messages(result, message, system_prompt)

                    if len(all_messages) > len(message):
                        error_patterns = [
                            "[ERROR: Session terminated]",
                            "[ERROR: Failed to connect to MCP server",
                            "[ERROR:",
                        ]
                        final_assistant_msgs = [
                            m for m in all_messages if m.get('role') == 'assistant' and m.get('content')
                        ]
                        has_error_response = False
                        if final_assistant_msgs:
                            last_content = final_assistant_msgs[-1].get('content', '')
                            for pattern in error_patterns:
                                if pattern in last_content and len(last_content.strip()) < 200:
                                    has_error_response = True
                                    print(f"  Agent response for item {prompt_id} contains MCP error: {last_content.strip()}")
                                    break

                        if has_error_response:
                            raise Exception(f"Agent response is an MCP error: {last_content.strip()}")

                        tool_count = len(mcp_servers) if mcp_servers else len(agent_config.get("tools", []))
                        source_type = "MCP servers" if mcp_servers else "Virtual Tools"
                        print(f"  OpenAI agent inference completed for item {prompt_id} with {tool_count} {source_type}")
                        item['messages'] = all_messages
                    else:
                        print(f"  OpenAI agent inference returned empty response for item {prompt_id}")
                        raise Exception("Agent returned empty response")

                finally:
                    for server_context in reversed(server_contexts):
                        try:
                            await server_context.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            print(f"  Warning: Failed to cleanup MCP server context: {cleanup_error}")

            except Exception as server_creation_error:
                print(f"  Failed to create MCP servers: {server_creation_error}")
                raise

        except Exception as e:
            print(f"  OpenAI agent inference failed for item {prompt_id}: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            if "async" in str(e).lower() or "context" in str(e).lower() or "sse" in str(e).lower():
                print(f"   This appears to be an async/context/MCP streaming error")
            raise e
    else:
        if args.agent:
            raise ValueError("Failed to create agent for this item")
        else:
            raise ValueError("No agent specified")

    return item


def extract_message_text(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        text_chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(part.get("text", ""))
        return "".join(text_chunks)
    return str(content)


def normalize_messages_for_completion(messages):
    if not messages:
        raise ValueError("No messages found")
    input_messages = messages
    if input_messages[0].get("role") == "system":
        input_messages = input_messages[1:]
    user_messages = [msg for msg in input_messages if msg.get("role") == "user"]
    if not user_messages:
        raise ValueError("No user messages found")
    latest_user_content = user_messages[-1]["content"]
    if input_messages[-1].get("role") != "user":
        raise ValueError("Last message is not a user message")
    return input_messages[:-1] + [{"role": "user", "content": latest_user_content}]


async def request_completion_async(messages, client):
    payload = {
        "model": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "messages": messages,
    }
    extra_body = {
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }
    if args.max_tokens is not None and args.max_tokens > 0:
        payload["max_tokens"] = args.max_tokens
    if extra_body:
        payload["extra_body"] = extra_body

    for attempt in range(args.max_retries):
        try:
            completion = await client.chat.completions.create(**payload)
            return extract_message_text(completion.choices[0].message.content).strip()
        except Exception as e:
            print(f"Request attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)

    return ""


async def process_single_item_direct_async(item, client):
    prompt_id = item.get("metadata", {}).get("prompt_id", "unknown")
    input_messages = normalize_messages_for_completion(item["messages"])

    print(f"  Using direct API for item {prompt_id}...")
    response = await request_completion_async(input_messages, client)
    if response is None:
        response = ""

    item["messages"] = input_messages + [{"role": "assistant", "content": response}]
    return item


# ================================================================
# Multi-turn: Item processing
# ================================================================

async def process_single_item_multiagent_async(item, api_key=None, profile=None):
    """Process a single item using the multi-agent Student-User loop."""
    metadata = item.get('metadata', {})
    prompt_id = metadata.get('prompt_id', 'unknown')

    question = metadata.get('question', '')
    if not question:
        messages = item.get('messages', [])
        user_msgs = [m for m in messages if m.get('role') == 'user']
        question = user_msgs[-1]['content'] if user_msgs else ''
    if not question:
        raise ValueError(f"No question found for item {prompt_id}")

    user_instructions = build_user_agent_instructions(metadata)

    student_config = create_student_agent_config(item, _make_client(), api_key, profile)
    if student_config is None:
        raise ValueError(f"Could not create student agent config for item {prompt_id}")
    if student_config.get("virtual_tool_diagnostics"):
        metadata["virtual_tool_diagnostics"] = student_config["virtual_tool_diagnostics"]

    user_client = _make_client()

    print(f"  Running multi-agent inference for item {prompt_id}...")

    mcp_server_contexts = []
    server_context_managers = []

    try:
        server_configs = student_config.get("mcp_servers_list", [])
        if server_configs:
            for server_config in server_configs:
                try:
                    mcp_server_ctx = MCPServerStreamableHttp(
                        name=server_config["name"],
                        params={
                            "url": server_config["url"],
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                            },
                            "timeout": server_config.get("timeout", 600.0),
                            "sse_read_timeout": server_config.get("sse_read_timeout", 600.0),
                            "terminate_on_close": server_config.get("terminate_on_close", False)
                        },
                        client_session_timeout_seconds=args.timeout,
                    )
                    mcp_server = await mcp_server_ctx.__aenter__()
                    mcp_server_contexts.append(mcp_server)
                    server_context_managers.append(mcp_server_ctx)
                except Exception as conn_error:
                    print(f"  Skipping MCP server '{server_config['name']}': {conn_error}")

            if not mcp_server_contexts and not student_config.get("tools"):
                raise Exception(f"All {len(server_configs)} MCP server(s) failed to connect")

        conversation_messages = await run_student_user_loop_async(
            student_agent_config=student_config,
            user_instructions=user_instructions,
            question=question,
            user_client=user_client,
            session_prefix=str(prompt_id),
            mcp_server_contexts=mcp_server_contexts if mcp_server_contexts else None,
        )

        if len(conversation_messages) > 1:
            print(f"  Multi-agent inference completed for item {prompt_id} ({len(conversation_messages)} messages)")
            item['messages'] = conversation_messages
        else:
            raise Exception("Multi-agent loop returned empty conversation")

    finally:
        for ctx in reversed(server_context_managers):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"  Warning: Failed to cleanup MCP server: {cleanup_error}")

    return item


def _multiagent_timeout_wrapper(item, api_key=None, profile=None):
    """Process a single item with timeout (multi-turn mode)."""
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')
    try:
        return asyncio.run(process_single_item_multiagent_async(item, api_key, profile))
    except Exception as e:
        print(f"Error processing item {prompt_id}: {str(e)}")
        print(traceback.format_exc())
        message = item["messages"]
        item['messages'] = message + [{"role": "assistant", "content": f"[ERROR: {str(e)}]"}]
        return item


# Apply the wrapt timeout decorator dynamically based on args.timeout
# This is needed because @timeout(args.timeout, use_signals=False) is evaluated
# at decoration time. We create it as a wrapped function.
def process_single_item_multiagent(item, api_key=None, profile=None):
    """Dispatch to timeout-wrapped multiagent processor."""
    decorated = wrapt_timeout(args.timeout, use_signals=False)(_multiagent_timeout_wrapper)
    return decorated(item, api_key, profile)


# ================================================================
# Multi-turn: DynamicProcessor
# ================================================================

class DynamicProcessor:
    def __init__(self, max_workers=None, checkpoint_every=16):
        self.max_workers = max_workers or (len(smithery_api_pool) if smithery_api_pool else 1)
        self.checkpoint_every = checkpoint_every
        self.processed_count = 0
        self.lock = threading.Lock()
        self.completed_items_list = []

    def process_single_item_with_fallback(self, item_data):
        item, item_index, api_key, profile = item_data
        prompt_id = item.get('metadata', {}).get('prompt_id', f'item_{item_index}')

        try:
            processed_item = process_single_item_multiagent(item, api_key, profile)
            return processed_item, item_index, True, None
        except Exception as e:
            print(f"  Multi-agent processing failed for item {prompt_id}: {str(e)}")
            message = item["messages"]
            item['messages'] = message + [
                {"role": "assistant", "content": f"[ERROR: Agent failed ({str(e)})]"}
            ]
            return item, item_index, False, f"Agent failed: {str(e)}"

    def process_items_dynamically(self, items_to_process, processed_dataset, checkpoint_file, progress_bar):
        completed_items = {}

        items_with_metadata = []
        for i, (item, original_index) in enumerate(items_to_process):
            api_key, profile = get_api_key_for_worker(i)
            items_with_metadata.append((item, original_index, api_key, profile))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {}
            for item_data in items_with_metadata:
                future = executor.submit(self.process_single_item_with_fallback, item_data)
                future_to_data[future] = item_data

            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    processed_item, original_index, success, error_msg = future.result()
                    completed_items[original_index] = processed_item
                    processed_dataset[original_index] = processed_item

                    with self.lock:
                        self.completed_items_list.append(processed_item)
                        self.processed_count += 1
                        progress_bar.update(1)

                        prompt_id = processed_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                        status = "[OK]" if success else "[FAIL]"
                        if error_msg:
                            print(f"{status} Completed item {prompt_id} (index {original_index}) - {error_msg}")
                        else:
                            print(f"{status} Completed item {prompt_id} (index {original_index})")

                        if self.processed_count % self.checkpoint_every == 0:
                            self._save_checkpoint_safely(checkpoint_file)

                except Exception as e:
                    item_data = future_to_data[future]
                    original_item, original_index, _, _ = item_data
                    prompt_id = original_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                    print(f"  Unexpected error processing item {prompt_id}: {str(e)}")

                    original_item['messages'] = original_item["messages"] + [
                        {"role": "assistant", "content": f"[UNEXPECTED_ERROR: {str(e)}]"}
                    ]
                    processed_dataset[original_index] = original_item

                    with self.lock:
                        self.completed_items_list.append(original_item)
                        self.processed_count += 1
                        progress_bar.update(1)

        with self.lock:
            if self.completed_items_list:
                self._save_checkpoint_safely(checkpoint_file, is_final=True)

        return len(completed_items)

    def _save_checkpoint_safely(self, checkpoint_file, is_final=False):
        try:
            existing_completed = []
            if os.path.exists(checkpoint_file):
                try:
                    existing_completed = load_dataset_from_file(checkpoint_file)
                    if not isinstance(existing_completed, list):
                        existing_completed = [existing_completed]
                    existing_completed = dedupe_checkpoint_items(existing_completed)
                except Exception as e:
                    print(f"  Warning: Could not load existing checkpoint: {e}")
                    existing_completed = []

            all_completed = existing_completed + self.completed_items_list
            all_completed_sorted = dedupe_checkpoint_items(all_completed)
            safe_save_checkpoint(all_completed_sorted, checkpoint_file, convert_to_jsonl=False)

            checkpoint_type = "Final" if is_final else "Periodic"
            print(f"  {checkpoint_type} checkpoint saved: {len(all_completed_sorted)} items")
            self.completed_items_list = []
        except Exception as e:
            print(f"  Error saving checkpoint: {e}")


# ================================================================
# Single-turn: generate_and_update (async)
# ================================================================

async def process_index_async(index, processed_dataset, direct_client, semaphore):
    async with semaphore:
        api_key, profile = get_api_key_for_worker(index)
        current_item = copy.deepcopy(processed_dataset[index])
        prompt_id = current_item.get("metadata", {}).get("prompt_id", f"item_{index}")

        try:
            if args.agent:
                processed_item = await asyncio.wait_for(
                    process_single_item_agent_async(current_item, api_key, profile),
                    timeout=args.timeout,
                )
            else:
                processed_item = await asyncio.wait_for(
                    process_single_item_direct_async(current_item, direct_client),
                    timeout=args.timeout,
                )
            print(f"  Completed item {prompt_id} (index {index})")
            return index, processed_item
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                error_text = f"Timed out after {args.timeout}s"
            else:
                error_text = str(e)
            print(f"  Failed item {prompt_id} (index {index}): {error_text}")
            return index, build_error_item(current_item, error_text)


async def generate_and_update_single(dataset, direct_client, checkpoint_dir):
    """Single-turn: process dataset items with async concurrency and per-item checkpoints."""
    processed_dataset = copy.deepcopy(dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    completed_indices = load_item_checkpoints(processed_dataset, checkpoint_dir)
    if completed_indices:
        print(
            f"Loaded {len(completed_indices)} completed item checkpoints from {checkpoint_dir}."
        )

    pending_indices = [idx for idx in range(len(processed_dataset)) if idx not in completed_indices]
    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)

    if not pending_indices:
        print("No remaining items to process.")
    else:
        print(f"Processing {len(pending_indices)} items with max concurrency {max_workers}.")
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            asyncio.create_task(process_index_async(idx, processed_dataset, direct_client, semaphore))
            for idx in pending_indices
        ]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating completions"):
            index, processed_item = await task
            processed_dataset[index] = processed_item
            save_item_checkpoint(index, processed_item, checkpoint_dir)

    generation_params = build_generation_params(max_workers)
    processed_dataset = add_generation_config_to_metadata(
        processed_dataset, model_abbreviation, generation_params,
    )
    return sort_dataset_by_row_id(processed_dataset)


async def run_trial_single(target_dataset, saved_file, checkpoint_dir):
    """Single-turn: run one trial."""
    direct_client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        updated_dataset = await generate_and_update_single(target_dataset, direct_client, checkpoint_dir)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"Final dataset saved to {saved_file}.")
    finally:
        await direct_client.close()


# ================================================================
# Multi-turn: generate_and_update (sync + ThreadPoolExecutor)
# ================================================================

def generate_and_update_multi(dataset, checkpoint_file):
    """Multi-turn: process dataset items with ThreadPoolExecutor and checkpoint file."""
    processed_dataset = copy.deepcopy(dataset)

    generation_params = build_generation_params(
        args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)
    )

    items_to_process = []
    completed_count = 0

    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_dataset_from_file(checkpoint_file)
            if not isinstance(checkpoint_data, list):
                checkpoint_data = [checkpoint_data]
            checkpoint_data = dedupe_checkpoint_items(checkpoint_data)
            print(f"Checkpoint file found with {len(checkpoint_data)} completed items.")

            completed_lookup = {}
            for completed_item in checkpoint_data:
                key = get_checkpoint_identity(completed_item)
                if key is not None:
                    completed_lookup[key] = completed_item

            completed_count = len(completed_lookup)

            for i, item in enumerate(processed_dataset):
                item_key = get_checkpoint_identity(item)
                if item_key is not None and item_key in completed_lookup:
                    processed_dataset[i] = completed_lookup[item_key]
                else:
                    items_to_process.append((item, i))

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh...")
            completed_count = 0
            for i in range(len(processed_dataset)):
                items_to_process.append((processed_dataset[i], i))
    else:
        print("No checkpoint found. Processing all items.")
        for i in range(len(processed_dataset)):
            items_to_process.append((processed_dataset[i], i))

    print(f"Total items in dataset: {len(processed_dataset)}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining to process: {len(items_to_process)}")

    if len(items_to_process) == 0:
        print("All items already processed!")
        return processed_dataset

    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)
    processor = DynamicProcessor(max_workers=max_workers, checkpoint_every=CHECKPOINT_EVERY)

    print(f"  Starting multi-agent processing with {max_workers} workers...")
    print(f"  Checkpoints every {CHECKPOINT_EVERY} items")
    print(f"  Item timeout: {args.timeout}s | Max user-agent turns: {args.user_max_turns}")

    with tqdm(total=len(items_to_process), desc="Processing items", unit="item", leave=True,
              dynamic_ncols=True,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as progress_bar:

        start_time = time()
        completed_count = processor.process_items_dynamically(
            items_to_process, processed_dataset, checkpoint_file, progress_bar
        )
        end_time = time()

        print(f"\n  Multi-agent processing completed!")
        print(f"  Items processed: {completed_count}/{len(items_to_process)}")
        print(f"  Total time: {end_time - start_time:.2f} seconds")

    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    processed_dataset_sorted = sort_dataset_by_row_id(processed_dataset)
    return processed_dataset_sorted


# ================================================================
# Lazy-load prompt templates (only for multi mode)
# ================================================================

USER_PROMPT_TEMPLATE = None
STUDENT_PROMPT_TEMPLATE = None


def _ensure_prompt_templates_loaded():
    global USER_PROMPT_TEMPLATE, STUDENT_PROMPT_TEMPLATE
    if USER_PROMPT_TEMPLATE is None:
        USER_PROMPT_TEMPLATE = load_user_prompt_template()
    if STUDENT_PROMPT_TEMPLATE is None:
        STUDENT_PROMPT_TEMPLATE = load_student_prompt_template()


# ================================================================
# main()
# ================================================================

async def main_single():
    """Entry point for single-turn mode."""
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be a positive integer.")

    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)
    pool_size = len(api_pool) if api_pool else 0
    effective_workers = args.max_workers or (pool_size if pool_size > 0 else 8)

    print("=" * 50)
    print("ASYNC SINGLE-TURN PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Workers: {effective_workers}")
    print(f"API pool size: {pool_size}")
    print(f"Timeout per item: {args.timeout} seconds")
    print(f"Checkpointing: One JSON file per completed item")
    print(f"Endpoint: {args.base_url}")
    print(f"Mode: {'Agent' if args.agent else 'Direct API'}")
    print(f"Tools: {args.tools}")
    print("=" * 50)

    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    if not isinstance(dataset, list):
        dataset = [dataset]

    total_rows = len(dataset)
    start_idx, requested_end_idx, end_idx = resolve_processing_range(total_rows)
    target_dataset = dataset[start_idx:end_idx]

    base_name = get_input_base_name(args.input_file)
    print(
        f"Dataset rows: {total_rows}. Requested range: "
        f"[{start_idx}, {requested_end_idx}). Effective range: [{start_idx}, {end_idx})."
    )
    print(f"Processing {len(target_dataset)} rows.")

    if args.num_trials == 1:
        saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx)
        print(f"Output file: {saved_file}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        await run_trial_single(target_dataset, saved_file, checkpoint_dir)
    else:
        for trial_idx in range(args.num_trials):
            saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx, trial_idx=trial_idx)
            print(f"Trial {trial_idx}: output={saved_file}")
            print(f"Trial {trial_idx}: checkpoints={checkpoint_dir}")
            await run_trial_single(target_dataset, saved_file, checkpoint_dir)

    print("Program execution completed.")


def main_multi():
    """Entry point for multi-turn mode."""
    _ensure_prompt_templates_loaded()

    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)

    pool_size = len(api_pool) if api_pool else 0
    effective_workers = args.max_workers or (pool_size if pool_size > 0 else 8)
    print("=" * 50)
    print("MULTI-AGENT PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Model (Student/User/VirtualTool): {args.model_path}")
    print(f"User max turns: {args.user_max_turns}")
    print(f"Max agent turns per student response: {args.max_turns}")
    print(f"Tool mode: {'Virtual' if args.virtual_tools else 'Real MCP'}")
    print(f"Endpoint: {args.base_url}")
    print(f"Workers: {effective_workers}")
    print(f"Timeout per item: {args.timeout} seconds")
    print("=" * 50)

    try:
        dataset = load_dataset_from_file(INPUT_FILE_NAME)
        if not isinstance(dataset, list):
            dataset = [dataset]

        total_rows = len(dataset)
        start_idx, requested_end_idx, end_idx = resolve_processing_range(total_rows)
        target_dataset = dataset[start_idx:end_idx]
        base_name = get_input_base_name(args.input_file)

        print(
            f"Dataset rows: {total_rows}. Requested range: "
            f"[{start_idx}, {requested_end_idx}). Effective range: [{start_idx}, {end_idx})."
        )
        print(f"Processing {len(target_dataset)} rows.")

        if args.num_trials == 1:
            saved_file, checkpoint_file = build_output_paths(base_name, start_idx, end_idx)
            updated_dataset = generate_and_update_multi(target_dataset, checkpoint_file)
            save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            print("Final dataset saved. Checkpoint removed.")
        else:
            for i in range(args.num_trials):
                saved_file, checkpoint_file = build_output_paths(base_name, start_idx, end_idx, trial_idx=i)
                updated_dataset = generate_and_update_multi(target_dataset, checkpoint_file)
                save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                print(f"Dataset for trial {i} saved.")

    finally:
        print("Program execution completed.")
        os._exit(0)


def main():
    if args.mode == "single":
        asyncio.run(main_single())
    else:
        main_multi()


if __name__ == "__main__":
    main()
