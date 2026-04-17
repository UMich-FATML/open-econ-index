# Open Econ Index

A benchmark for evaluating LLM agents on occupational tasks, built on the O\*NET taxonomy and WildChat conversations.

## Components

### `chat_index/` — Chat-to-O\*NET Task Mapping

Maps conversations from the [WildEnglishChats-1.6M](https://huggingface.co/datasets/umich-fatml/WildEnglishChats-1.6M) dataset to O\*NET occupational tasks using Qwen3-Embedding semantic search.

```bash
cd chat_index
python map_summaries_to_tasks.py --top_k 3
```

The combined dataset (WildChat conversations + O\*NET task mappings, including stable
task IDs) is available on HuggingFace as `umich-fatml/OpenEconIndex`.

**`OpenEconIndexAnalysis.ipynb`** — reproduces the paper figures (occupational
representation, depth of AI usage, top skills, work activity coverage, wage vs AI
usage). Loads the dataset directly from HuggingFace.

### `analysis/` — Tools Coverage Analysis

- **`check_cophenetic_correlation.py`** — Measures alignment between embedding distances and O\*NET work activity hierarchy (GWA → IWA → DWA → Task).
- **`match_commodities_to_servers.py`** — Matches O\*NET software commodities to Smithery MCP servers using semantic embeddings.

### `benchmark/` — Full Evaluation Pipeline

A 9-step pipeline for generating occupational task questions, running LLM agent trajectories, and evaluating them across 5 dimensions.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_generate_questions.py` / `step1_hero_generate.py` | Generate question prompts from O\*NET tasks |
| 2 | `step2_structured_completion.py` | Run structured LLM completions |
| 3 | `step3_process_completion.py` | Process and deduplicate completions |
| 4 | `step4_validate_and_convert.py` | Validate and convert to agent-ready format |
| 5 | `step5_agent.py` | Generate agent trajectories |
| 6 | `step6_prepare_eval.py` | Prepare evaluation prompts (all dimensions) |
| 7 | `step7_eval_completion.sh` / `step7_eval_endpoint.py` | Run LLM judge evaluation |
| 8 | `step8_process_eval_scores.py` / `step8_aggregate_eval_scores.py` | Process and aggregate scores |
| 9 | `step9_build_leaderboard.py` | Build leaderboard |

#### Agent Mode Toggles (Step 5)

`step5_agent.py` has two orthogonal toggles:

- **`--mode single | multi`** — single-turn (one user prompt → one agent response) or
  multi-turn (alternating Student–User loop up to `--user_max_turns` rounds).
- **`--tools virtual | real`** — simulate tool calls via an LLM (`VirtualToolBackend`)
  or actually call live Smithery MCP servers.

```bash
# Defaults: multi-turn + virtual tools (no external dependencies beyond OpenRouter)
python step5_agent.py --input_file ...

# Single-turn + virtual tools
python step5_agent.py --mode single --tools virtual --input_file ...

# Multi-turn + real MCP servers (see "Using real MCP servers" below)
python step5_agent.py --mode multi --tools real --input_file ...
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

3. **Unzipped MCP server metadata** — `benchmark/step5_agent.py` reads from
   `mcp_servers/smithery_mcp_servers_0210/` to know which servers/tools each
   question targets. Make sure you've unzipped it (see Setup below).

4. **Network access** — the agent opens streamable-HTTP connections to each
   server's `deploymentUrl` (read from its metadata JSON). Some Smithery-hosted
   servers are rate-limited or region-gated; individual failures are logged
   per-item and don't abort the run.

With these in place, `--tools real` behaves like `--tools virtual` but the
tool responses come from live servers instead of a simulating LLM.

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

# Unzip MCP server definitions (required for benchmark/step5_agent.py)
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
