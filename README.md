# Open Econ Index

A benchmark for evaluating LLM agents on occupational tasks, built on the O\*NET taxonomy and WildChat conversations.

## Components

### `chat_index/` — Chat-to-O\*NET Task Mapping

Two-stage mapping of conversations from [WildEnglishChats-1.6M](https://huggingface.co/datasets/umich-fatml/WildEnglishChats-1.6M)
onto O\*NET occupational tasks:

1. **Embedding retrieval** — `map_summaries_to_tasks.py` embeds conversation
   summaries with Qwen3-Embedding and assigns the top-k closest O\*NET tasks
   by cosine similarity. Output: `summaries2tasks.json` (unfiltered, top-3 per chat).
2. **LLM filtering** — `filter_tasks.py` asks an LLM (via OpenRouter) which of
   the candidate tasks are genuinely represented by the conversation, dropping
   spurious matches. Output: `summaries2tasks-filtered.json`.

```bash
cd chat_index

# Stage 1: embedding-based candidate retrieval
python map_summaries_to_tasks.py --top_k 3

# Stage 2: LLM filtering (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-key"
python filter_tasks.py --model meta-llama/llama-3.1-8b-instruct
```

`filter_tasks.py` defaults to US-only conversations (uses `us-hash.json`).
Pass `--no_us_filter` to process all chats, `--limit N` for a quick test.

The combined dataset (WildChat conversations + O\*NET task mappings, including stable
task IDs) is available on HuggingFace as `umich-fatml/OpenEconIndex`.

**`OpenEconIndexAnalysis.ipynb`** — reproduces the paper figures (occupational
representation, depth of AI usage, top skills, work activity coverage, wage vs AI
usage). Loads the dataset directly from HuggingFace.

### `analysis/` — Tools Coverage Analysis

- **`check_cophenetic_correlation.py`** — Measures alignment between embedding distances and O\*NET work activity hierarchy (GWA → IWA → DWA → Task).
- **`match_commodities_to_servers.py`** — Matches O\*NET software commodities to Smithery MCP servers using semantic embeddings.

### `benchmark/` — Full Evaluation Pipeline

An 8-stage pipeline that generates occupational task questions, runs LLM agent
trajectories, and evaluates them across 5 dimensions. Numbered `stepN_…` files
are pipeline stages; the agent / virtual-tool / structured-completion / eval
endpoints are reusable components those stages call.

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `step1_generate_questions.py` / `step1_hero_generate.py` | Generate question prompts from O\*NET tasks |
| 2 | `structured_completion.py` | Run structured LLM completions |
| 3 | `step3_process_completion.py` | Process and deduplicate completions |
| 4 | `step4_validate_and_convert.py` | Validate and convert to agent-ready format |
| 5 | `agent.py` | Generate agent trajectories (uses `virtual_tools.py` when `--tools virtual`) |
| 6 | `step6_prepare_eval.py` | Prepare evaluation prompts (all dimensions) |
| 7 | `step7_eval_completion.sh` | Run LLM judge evaluation (calls `eval_endpoint.py` per dimension) |
| 8 | `step8_process_eval_scores.py` / `step8_aggregate_eval_scores.py` | Process and aggregate scores |

#### Agent Mode Toggles (Stage 5)

`agent.py` has two orthogonal toggles:

- **`--mode single | multi`** — single-turn (one user prompt → one agent response) or
  multi-turn (alternating Student–User loop up to `--user_max_turns` rounds).
- **`--tools virtual | real`** — simulate tool calls via an LLM (`virtual_tools.VirtualToolBackend`)
  or actually call live Smithery MCP servers.

```bash
# Defaults: multi-turn + virtual tools (no external dependencies beyond an LLM API)
python agent.py --input_file ...

# Single-turn + virtual tools
python agent.py --mode single --tools virtual --input_file ...

# Multi-turn + real MCP servers (see "Using real MCP servers" below)
python agent.py --mode multi --tools real --input_file ...
```

##### Using real MCP servers (`--tools real`)

Real mode connects to live Smithery-hosted MCP servers via streamable HTTP.
You'll need:

1. **A Smithery account and API key(s)** — sign up at [smithery.ai](https://smithery.ai/),
   then copy your API key from the dashboard. Multiple keys are supported and
   rotated across workers for higher throughput.

2. **A `benchmark/smithery_api_pool.json` file** — based on
   `smithery_api_pool.json.example`, populate it with one or more keys:
   ```json
   [
     {"name": "pool-key-1", "api_key": "sk-..."},
     {"name": "pool-key-2", "api_key": "sk-..."}
   ]
   ```
   This file is gitignored. Never commit real keys.

3. **Unzipped MCP server metadata** — `benchmark/agent.py` reads from
   `mcp_servers/smithery_mcp_servers_0210/` to know which servers/tools each
   question targets. Make sure you've unzipped it (see Setup below).

4. **Network access** — the agent opens streamable-HTTP connections to each
   server's `deploymentUrl` (read from its metadata JSON). Some Smithery-hosted
   servers are rate-limited or region-gated; individual failures are logged
   per-item and don't abort the run.

With these in place, `--tools real` behaves like `--tools virtual` but the
tool responses come from live servers instead of a simulating LLM.

#### Choosing the LLM Backend (OpenRouter vs. local vLLM)

Three pipeline stages call an LLM: stage 2 (`structured_completion.py`),
stage 5 (`agent.py`), and stage 7 (`eval_endpoint.py`, invoked by
`step7_eval_completion.sh`). All three accept the same toggle pattern: an
`--engine` choice plus a `--base_url` / `--api_key` pair.

**OpenRouter (hosted models, no local GPU required):**

```bash
# Stage 2
python structured_completion.py \
  --model_name moonshotai/kimi-k2.5 \
  --engine openrouter_api \
  --openrouter_api_key "$OPENROUTER_API_KEY" \
  --input_file ...

# Stage 5
python agent.py \
  --model_path moonshotai/kimi-k2.5 \
  --base_url https://openrouter.ai/api/v1 \
  --api_key "$OPENROUTER_API_KEY" \
  --input_file ...

# Stage 7 (via shell wrapper)
bash step7_eval_completion.sh \
  --input_file ... \
  --engine openrouter_api \
  --model_path openai/gpt-oss-120b
```

**Local vLLM (run `vllm serve <model> --port 8000` first):**

```bash
# Stage 2
python structured_completion.py \
  --model_name <local-model> \
  --engine vllm_api \
  --base_url http://localhost:8000/v1 \
  --input_file ...

# Stage 5
python agent.py \
  --model_path <local-model> \
  --base_url http://localhost:8000/v1 \
  --api_key EMPTY \
  --input_file ...

# Stage 7 (via shell wrapper)
bash step7_eval_completion.sh \
  --input_file ... \
  --engine vllm_api \
  --base_url http://localhost:8000/v1 \
  --model_path <local-model>
```

`run_pipeline.sh` defaults to OpenRouter; override the relevant flags or env
vars (`OPENROUTER_URL`, etc.) to switch to vLLM.

#### Eval Dimensions (Step 6)

```bash
# Prepare all 5 dimensions at once
python step6_prepare_eval.py --dimensions all --input_file ...

# Or select specific dimensions
python step6_prepare_eval.py --dimensions tool_call,grounding --input_file ...
```

Dimensions: `tool_call`, `workflow_completion`, `grounding`, `followup_quality`, `autonomy`

#### Full Pipeline

```bash
cd benchmark
export OPENROUTER_API_KEY="your-key"
bash run_pipeline.sh moonshotai/kimi-k2.5
```

## Setup

A conda environment is required — installs into a system Python or venv have been
unreliable for this stack (faiss + torch + openai-agents combination).

```bash
conda create -n oei python=3.11
conda activate oei
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Unzip MCP server definitions (required for benchmark/agent.py)
cd mcp_servers && unzip smithery_mcp_servers_0210.zip && cd ..
```

All subsequent commands assume you have `oei` activated:

```bash
conda activate oei
```

## Data

- **`onet_db/`** — O\*NET v30.1 database (43 text files)
- **`mcp_servers/`** — Smithery MCP server definitions (1520 JSON files)
- **`benchmark/prompts/`** — Prompt templates for question generation and evaluation
- **`benchmark/tasks_to_smithery_servers.jsonl`** — O\*NET task-to-MCP server mappings

## License

TBD
