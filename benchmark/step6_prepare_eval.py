#!/usr/bin/env python3
"""
Step 6: Unified Eval Preparation

Merges five separate eval-prep scripts into a single CLI:
  - tool_call            (from step4.1)
  - workflow_completion   (from step4.2)
  - grounding             (from step4.3)
  - followup_quality      (from step4.15)
  - autonomy              (from step4.16)

Usage:
  python step6_prepare_eval.py \
    --input_file path/to/results.jsonl \
    --answer_key_file path/to/answer_key.jsonl \
    --dimensions tool_call,workflow_completion,followup_quality,autonomy,grounding

  python step6_prepare_eval.py \
    --input_file path/to/results.jsonl \
    --dimensions all
"""

import os
import json
import re
import argparse
from collections import defaultdict
from tqdm import tqdm
from utils import (
    load_prompt_template,
    derive_answer_key_path,
    extract_raw_error,
    condense_trajectory,
    extract_final_response,
    extract_tool_evidence,
    extract_assistant_content,
    get_trajectory_status,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Prompt template loading (lazy, so missing templates only error when used)
# ---------------------------------------------------------------------------

_PROMPT_CACHE = {}


def _get_prompt(template_name):
    """Load and cache a prompt template by name."""
    if template_name not in _PROMPT_CACHE:
        _PROMPT_CACHE[template_name] = load_prompt_template(
            os.path.join(SCRIPT_DIR, template_name)
        )
    return _PROMPT_CACHE[template_name]


# ===================================================================
# TOOL CALL dimension  (from step4.1)
# ===================================================================

def _parse_tool_args(raw_args):
    """Parse tool call arguments from string or dict."""
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return raw_args
    return raw_args


def extract_all_tool_calls(messages, tool_name):
    """
    Extract all calls to *tool_name* from the trajectory, chronologically.
    Returns list of argument dicts.
    """
    results = []
    for msg in messages:
        # OpenAI 'tool_calls' list
        if 'tool_calls' in msg and msg['tool_calls']:
            for tc in msg['tool_calls']:
                func = tc.get('function', {})
                if func.get('name') == tool_name:
                    results.append(_parse_tool_args(func.get('arguments', '{}')))

        # Legacy 'function_call' dict
        if 'function_call' in msg and msg['function_call']:
            func = msg['function_call']
            if func.get('name') == tool_name:
                results.append(_parse_tool_args(func.get('arguments', '{}')))

    return results


def build_single_eval_prompt(tool_name, expected_args, actual_args):
    """Build prompt for a single expected-vs-actual tool call comparison."""
    return (
        f"Please evaluate the following tool call pair.\n\n"
        f"EXPECTED Tool: {tool_name}\n"
        f"EXPECTED Arguments: {json.dumps(expected_args, indent=2)}\n\n"
        f"ACTUAL Tool: {tool_name}\n"
        f"ACTUAL Arguments: {json.dumps(actual_args, indent=2)}"
    )


def build_multi_eval_prompt(tool_name, expected_args_list, actual_args_list):
    """Build prompt for comparing multiple expected calls vs multiple actual calls."""
    parts = [
        f"Please evaluate the following MULTI-CALL tool comparison for tool: {tool_name}\n",
        f"The answer key expects {len(expected_args_list)} call(s) to this tool. "
        f"The agent made {len(actual_args_list)} call(s).\n",
    ]

    parts.append("## EXPECTED Calls (order does NOT matter):")
    for idx, args in enumerate(expected_args_list, 1):
        parts.append(f"\nExpected Call {idx}:\n{json.dumps(args, indent=2)}")

    parts.append("\n## ACTUAL Calls (order does NOT matter):")
    for idx, args in enumerate(actual_args_list, 1):
        parts.append(f"\nActual Call {idx}:\n{json.dumps(args, indent=2)}")

    return "\n".join(parts)


def build_tool_call_prompts(trajectories, answer_keys_map):
    """
    Build eval prompts and auto-scores for the tool_call dimension.
    Returns (eval_records, auto_score_records).
    """
    eval_system_prompt = _get_prompt('prompts/evaluator.md')
    eval_prompts = []
    auto_scores = []

    for i, run in enumerate(trajectories):
        row_id = run.get('metadata', {}).get('row_id', i)
        messages = run.get('messages', [])
        expected_chain = answer_keys_map.get(row_id)

        if not expected_chain:
            continue

        # --- Check for global failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        # Group expected tools by name
        expected_by_name = defaultdict(list)
        for entry in expected_chain:
            name = entry.get('tool')
            expected_by_name[name].append(entry.get('arguments'))

        unique_tool_names = list(expected_by_name.keys())

        if is_failed:
            for tool_name in unique_tool_names:
                auto_scores.append({
                    "metadata": {
                        "original_row_id": row_id,
                        "eval_dimension": "tool_call",
                        "tool_name": tool_name,
                        "expected_count": len(expected_by_name[tool_name]),
                        "actual_count": 0,
                        "scenario_total_unique_tools": len(unique_tool_names),
                    },
                    "score": 0,
                    "reasoning": f"Scenario Failed. Raw Error: {raw_error_string}",
                    "error_type": raw_error_string,
                    "status": "scenario_failed"
                })
            continue

        # --- Process tools grouped by name ---
        for tool_name in unique_tool_names:
            expected_args_list = expected_by_name[tool_name]
            n_expected = len(expected_args_list)

            all_actual = extract_all_tool_calls(messages, tool_name)

            meta = {
                "original_row_id": row_id,
                "eval_dimension": "tool_call",
                "tool_name": tool_name,
                "expected_count": n_expected,
                "actual_count": len(all_actual),
                "scenario_total_unique_tools": len(unique_tool_names),
            }

            if not all_actual:
                # No calls to this tool at all
                auto_scores.append({
                    "metadata": meta,
                    "score": 0,
                    "reasoning": "Tool was not called by the agent.",
                    "error_type": "MISSING_TOOL_CALL",
                    "status": "missing_tool"
                })
            elif n_expected == 1:
                # Single expected call -- use last actual
                user_content = build_single_eval_prompt(
                    tool_name, expected_args_list[0], all_actual[-1]
                )
                eval_prompts.append({
                    "messages": [
                        {"role": "system", "content": eval_system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "metadata": meta
                })
            else:
                # Multiple expected calls -- compare lists (take N most recent actual)
                actual_subset = all_actual[-n_expected:]
                user_content = build_multi_eval_prompt(
                    tool_name, expected_args_list, actual_subset
                )
                eval_prompts.append({
                    "messages": [
                        {"role": "system", "content": eval_system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "metadata": meta
                })

    return eval_prompts, auto_scores


# ===================================================================
# WORKFLOW COMPLETION dimension  (from step4.2)
# ===================================================================

def build_workflow_eval_prompt(question, cross_tool_workflow, tool_analysis,
                               condensed_trajectory, final_response):
    """Build the user-content portion of a workflow completion evaluation prompt."""
    parts = [
        f"## Original Question\n{question}",
        f"\n## Expected Cross-Tool Workflow\n{cross_tool_workflow}",
    ]

    if tool_analysis:
        parts.append(f"\n## Tool Analysis (Supporting Context)\n{tool_analysis}")

    parts.append(f"\n## Agent Trajectory (Condensed)\n{condensed_trajectory}")
    parts.append(f"\n## Agent's Final Response\n{final_response if final_response else '(No final response produced)'}")

    return "\n".join(parts)


def build_workflow_prompts(trajectories, answer_keys_map):
    """
    Build eval prompts and auto-scores for the workflow_completion dimension.
    Returns (eval_records, auto_score_records).
    """
    eval_system_prompt = _get_prompt('prompts/eval_workflow_completion.md')
    eval_prompts = []
    auto_scores = []

    for i, run in enumerate(trajectories):
        metadata = run.get('metadata', {})
        row_id = metadata.get('row_id', i)
        messages = run.get('messages', [])

        ak_meta = answer_keys_map.get(row_id, {}).get('metadata', {})

        question = metadata.get('question', ak_meta.get('question', ''))
        cross_tool_workflow = metadata.get('cross_tool_workflow', ak_meta.get('cross_tool_workflow', ''))
        tool_analysis = metadata.get('tool_analysis', ak_meta.get('tool_analysis', ''))

        # --- Check for missing workflow reference ---
        if not cross_tool_workflow:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "workflow_completion",
                },
                "score": None,
                "reasoning": "No cross_tool_workflow reference found in metadata.",
                "status": "missing_reference",
            })
            continue

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "workflow_completion",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Build eval prompt (one per scenario) ---
        condensed = condense_trajectory(messages)
        final_response = extract_final_response(messages)

        user_content = build_workflow_eval_prompt(
            question=question,
            cross_tool_workflow=cross_tool_workflow,
            tool_analysis=tool_analysis,
            condensed_trajectory=condensed,
            final_response=final_response,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "workflow_completion",
            },
        })

    return eval_prompts, auto_scores


# ===================================================================
# GROUNDING dimension  (from step4.3)
# ===================================================================

def format_tool_evidence(evidence_list):
    """Format extracted tool evidence into readable text for the judge."""
    if not evidence_list:
        return "(No tool calls were made)"

    parts = []
    for idx, ev in enumerate(evidence_list, 1):
        parts.append(f"### Tool Call {idx}: {ev['tool_name']}")
        args_display = ev['arguments_summary']
        if len(args_display) > 500:
            args_display = args_display[:500] + " ... (truncated)"
        parts.append(f"**Arguments**: {args_display}")

        output_display = ev['output'] if ev['output'] else "(no output received)"
        if len(output_display) > 1500:
            output_display = output_display[:1500] + " ... (truncated)"
        parts.append(f"**Output**: {output_display}")
        parts.append("")  # blank line separator

    return "\n".join(parts)


def build_grounding_eval_prompt(question, tool_evidence_text, assistant_content):
    """Build the user-content portion of a grounding evaluation prompt."""
    parts = [
        f"## Original Question\n{question}",
        f"\n## Tool Calls and Outputs (Evidence Base)\n{tool_evidence_text}",
        f"\n## Assistant Messages (Claims to Evaluate)\n{assistant_content if assistant_content else '(No assistant messages produced)'}",
    ]
    return "\n".join(parts)


def build_grounding_prompts(trajectories, answer_keys_map):
    """
    Build eval prompts and auto-scores for the grounding dimension.
    Returns (eval_records, auto_score_records).
    """
    eval_system_prompt = _get_prompt('prompts/eval_grounding.md')
    eval_prompts = []
    auto_scores = []

    for i, run in enumerate(trajectories):
        metadata = run.get('metadata', {})
        row_id = metadata.get('row_id', i)
        messages = run.get('messages', [])

        ak_meta = answer_keys_map.get(row_id, {}).get('metadata', {})
        question = metadata.get('question', ak_meta.get('question', ''))

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "grounding",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Check for empty assistant content ---
        assistant_content = extract_assistant_content(messages)

        if not assistant_content.strip():
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "grounding",
                },
                "score": 0,
                "reasoning": "Agent produced no assistant text content.",
                "error_type": "NO_ASSISTANT_CONTENT",
                "status": "no_content",
            })
            continue

        # --- Build eval prompt ---
        tool_evidence = extract_tool_evidence(messages)
        tool_evidence_text = format_tool_evidence(tool_evidence)

        user_content = build_grounding_eval_prompt(
            question=question,
            tool_evidence_text=tool_evidence_text,
            assistant_content=assistant_content,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "grounding",
            },
        })

    return eval_prompts, auto_scores


# ===================================================================
# FOLLOWUP QUALITY dimension  (from step4.15)
# ===================================================================

def format_withheld_info(withheld_info):
    """Format withheld_info list for the eval prompt."""
    if not withheld_info or not isinstance(withheld_info, list):
        return "(None provided)"
    lines = []
    for item in withheld_info:
        param = item.get("parameter", "")
        desc = item.get("description", "")
        value = item.get("value", "")
        lines.append(f"- **{param}**: {desc} -> correct value: `{value}`")
    return "\n".join(lines)


def format_target_followup_questions(questions):
    """Format target follow-up questions for the eval prompt."""
    if not questions or not isinstance(questions, list):
        return "(None provided)"
    return "\n".join(f"- {q}" for q in questions)


def build_followup_eval_prompt(question, withheld_info, target_followup_questions,
                                condensed_trajectory):
    """Build the user-content portion of a follow-up quality evaluation prompt."""
    parts = [
        f"## Original Request (with deliberate omissions)\n{question}",
        f"\n## Withheld Information (parameters the agent needed to ask about)\n{format_withheld_info(withheld_info)}",
        f"\n## Target Follow-up Questions (what the agent should have asked)\n{format_target_followup_questions(target_followup_questions)}",
        f"\n## Agent Trajectory (Condensed)\n{condensed_trajectory}",
    ]
    return "\n".join(parts)


def build_followup_prompts(trajectories, answer_keys_map):
    """
    Build eval prompts and auto-scores for the followup_quality dimension.
    Returns (eval_records, auto_score_records).
    """
    eval_system_prompt = _get_prompt('prompts/eval_followup_quality.md')
    eval_prompts = []
    auto_scores = []

    for i, run in enumerate(trajectories):
        metadata = run.get('metadata', {})
        row_id = metadata.get('row_id', i)
        messages = run.get('messages', [])

        ak_meta = answer_keys_map.get(row_id, {}).get('metadata', {})

        question = metadata.get('question', ak_meta.get('question', ''))
        withheld_info = metadata.get('withheld_info', ak_meta.get('withheld_info', None))
        target_followup_questions = metadata.get(
            'target_followup_questions',
            ak_meta.get('target_followup_questions', None)
        )

        # --- Auto-score scenarios without withheld_info (not withheld-info scenarios) ---
        if not withheld_info or not isinstance(withheld_info, list) or len(withheld_info) == 0:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "followup_quality",
                },
                "score": None,
                "reasoning": "No withheld_info found in metadata -- not a withheld-info scenario.",
                "status": "missing_reference",
            })
            continue

        # Ensure we have at least some target questions (use empty list if missing)
        if not target_followup_questions or not isinstance(target_followup_questions, list):
            target_followup_questions = []

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)
        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "followup_quality",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Build eval prompt ---
        condensed = condense_trajectory(messages)

        user_content = build_followup_eval_prompt(
            question=question,
            withheld_info=withheld_info,
            target_followup_questions=target_followup_questions,
            condensed_trajectory=condensed,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "followup_quality",
            },
        })

    return eval_prompts, auto_scores


# ===================================================================
# AUTONOMY dimension  (from step4.16)
# ===================================================================

STEERING_PATTERNS = [
    r"\btry\b",
    r"\bre-?run\b",
    r"\bagain\b",
    r"\bredo\b",
    r"\bwrong\b",
    r"\bstuck\b",
    r"\bproper\b",
    r"\bdid you\b",
    r"\bcan you\b",
    r"\byou should\b",
    r"\bgo ahead\b",
    r"\bbefore\b",
    r"\bneed to\b",
]


def compute_autonomy_telemetry(messages):
    """Extract deterministic telemetry features to aid autonomy judging."""
    traj_status = get_trajectory_status(messages)

    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    user_followups = user_msgs[1:] if len(user_msgs) > 1 else []

    steering_turns = 0
    rerun_turns = 0
    for content in user_followups:
        lower = str(content).lower()
        if any(re.search(pattern, lower) for pattern in STEERING_PATTERNS):
            steering_turns += 1
        if re.search(r"\bre-?run\b|\bagain\b|\bredo\b|\bproper\b", lower):
            rerun_turns += 1

    assistant_tool_call_messages = 0
    for msg in messages:
        if msg.get("role") == "assistant" and (msg.get("tool_calls") or msg.get("function_call")):
            assistant_tool_call_messages += 1

    telemetry = {
        "user_turn_count": traj_status["user_turn_count"],
        "assistant_turn_count": traj_status["assistant_turn_count"],
        "is_multi_turn": traj_status["is_multi_turn"],
        "ended_with_end_conversation": traj_status["ended_with_end_conversation"],
        "turn_expired": traj_status["turn_expired"],
        "trailing_tool_call": traj_status["trailing_tool_call"],
        "user_steering_turns": steering_turns,
        "explicit_rerun_turns": rerun_turns,
        "assistant_tool_call_messages": assistant_tool_call_messages,
    }
    return telemetry


def format_telemetry(telemetry):
    """Render telemetry for judge prompt."""
    return "\n".join([
        f"- User turns: {telemetry['user_turn_count']}",
        f"- Assistant turns: {telemetry['assistant_turn_count']}",
        f"- Multi-turn trajectory: {telemetry['is_multi_turn']}",
        f"- Ended with <END_CONVERSATION>: {telemetry['ended_with_end_conversation']}",
        f"- Turn-expired terminal shape: {telemetry['turn_expired']}",
        f"- Trailing tool-call terminal shape: {telemetry['trailing_tool_call']}",
        f"- User steering turns (heuristic): {telemetry['user_steering_turns']}",
        f"- Explicit rerun/correction turns: {telemetry['explicit_rerun_turns']}",
        f"- Assistant tool-call messages: {telemetry['assistant_tool_call_messages']}",
    ])


def build_autonomy_eval_prompt(question, cross_tool_workflow, condensed_trajectory,
                               final_response, telemetry, tool_analysis="",
                               withheld_info=None, target_followup_questions=None):
    """Build the user-content portion of an autonomy evaluation prompt."""
    parts = [
        f"## Original Question\n{question}",
        f"\n## Expected Cross-Tool Workflow\n{cross_tool_workflow}",
    ]

    if tool_analysis:
        parts.append(f"\n## Tool Analysis (Supporting Context)\n{tool_analysis}")

    parts.extend([
        (
            "\n## Withheld Information (If Any)\n"
            f"{format_withheld_info(withheld_info)}"
        ),
        (
            "\n## Target Follow-up Questions (Reference, Not Exhaustive)\n"
            f"{format_target_followup_questions(target_followup_questions)}"
        ),
        f"\n## Agent Trajectory (Condensed)\n{condensed_trajectory}",
        f"\n## Agent Final Response\n{final_response if final_response else '(No final response produced)'}",
        f"\n## Telemetry (Deterministic Signals)\n{format_telemetry(telemetry)}",
    ])
    return "\n".join(parts)


def build_autonomy_prompts(trajectories, answer_keys_map):
    """
    Build eval prompts and auto-scores for the autonomy dimension.
    Returns (eval_records, auto_score_records).
    """
    eval_system_prompt = _get_prompt('prompts/eval_autonomy.md')
    eval_prompts = []
    auto_scores = []

    for i, run in enumerate(trajectories):
        metadata = run.get('metadata', {})
        row_id = metadata.get('row_id', i)
        messages = run.get('messages', [])

        ak_meta = answer_keys_map.get(row_id, {}).get('metadata', {})

        question = metadata.get('question', ak_meta.get('question', ''))
        cross_tool_workflow = metadata.get('cross_tool_workflow', ak_meta.get('cross_tool_workflow', ''))
        tool_analysis = metadata.get('tool_analysis', ak_meta.get('tool_analysis', ''))
        withheld_info = metadata.get('withheld_info', ak_meta.get('withheld_info', None))
        target_followup_questions = metadata.get(
            'target_followup_questions',
            ak_meta.get('target_followup_questions', None)
        )

        telemetry = compute_autonomy_telemetry(messages)
        traj_status = get_trajectory_status(messages)

        base_meta = {
            "original_row_id": row_id,
            "eval_dimension": "autonomy",
            "is_multi_turn": traj_status["is_multi_turn"],
            "turn_expired": traj_status["turn_expired"],
        }

        # --- Missing workflow reference ---
        if not cross_tool_workflow:
            auto_scores.append({
                "metadata": base_meta,
                "score": None,
                "reasoning": "No cross_tool_workflow reference found in metadata.",
                "telemetry": telemetry,
                "status": "missing_reference",
            })
            continue

        # --- Catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)
        if is_failed:
            auto_scores.append({
                "metadata": base_meta,
                "score": 1,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "telemetry": telemetry,
                "status": "scenario_failed",
            })
            continue

        # --- Turn-expired auto-score ---
        if traj_status["turn_expired"]:
            auto_scores.append({
                "metadata": base_meta,
                "score": 1,
                "reasoning": "Trajectory appears turn-expired / unresolved before explicit conversation end.",
                "telemetry": telemetry,
                "status": "turn_expired",
            })
            continue

        # --- Build eval prompt ---
        condensed = condense_trajectory(messages)
        final_response = extract_final_response(messages)

        user_content = build_autonomy_eval_prompt(
            question=question,
            cross_tool_workflow=cross_tool_workflow,
            condensed_trajectory=condensed,
            final_response=final_response,
            telemetry=telemetry,
            tool_analysis=tool_analysis,
            withheld_info=withheld_info,
            target_followup_questions=target_followup_questions,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "metadata": base_meta,
        })

    return eval_prompts, auto_scores


# ===================================================================
# Dimension registry
# ===================================================================

ALL_DIMENSIONS = [
    "tool_call",
    "workflow_completion",
    "grounding",
    "followup_quality",
    "autonomy",
]

DIMENSION_REGISTRY = {
    "tool_call": {
        "prompt_template": "prompts/evaluator.md",
        "builder": build_tool_call_prompts,
    },
    "workflow_completion": {
        "prompt_template": "prompts/eval_workflow_completion.md",
        "builder": build_workflow_prompts,
    },
    "grounding": {
        "prompt_template": "prompts/eval_grounding.md",
        "builder": build_grounding_prompts,
    },
    "followup_quality": {
        "prompt_template": "prompts/eval_followup_quality.md",
        "builder": build_followup_prompts,
    },
    "autonomy": {
        "prompt_template": "prompts/eval_autonomy.md",
        "builder": build_autonomy_prompts,
    },
}


# ===================================================================
# CLI & main loop
# ===================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Unified Eval Preparation: build eval prompts for one or more dimensions."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the agent trajectory results file (*.jsonl).",
    )
    parser.add_argument(
        "--answer_key_file", type=str, default=None,
        help="Path to the answer key file. If omitted, attempts to derive it from input_file.",
    )
    parser.add_argument(
        "--dimensions", type=str, default="all",
        help=(
            "Comma-separated list of dimensions to prepare, or 'all'. "
            f"Choices: {', '.join(ALL_DIMENSIONS)}"
        ),
    )
    return parser.parse_args()


def load_trajectories(input_file):
    """Load trajectory records from a JSONL file."""
    print(f"Loading trajectories from: {input_file}")
    with open(input_file, 'r') as f:
        trajectories = [json.loads(line) for line in f]
    print(f"  Loaded {len(trajectories)} scenarios")
    return trajectories


def load_answer_keys(input_file, answer_key_file):
    """
    Load the answer key map.

    For the tool_call dimension the map values are the answer_key lists.
    For every other dimension the map values are the full record dicts (with metadata).

    Returns (tool_call_answer_keys_map, full_answer_keys_map).
    """
    answer_key_path = answer_key_file or derive_answer_key_path(input_file)

    tool_call_map = {}
    full_map = {}

    if answer_key_path and os.path.exists(answer_key_path):
        print(f"Loading answer key from: {answer_key_path}")
        with open(answer_key_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                rid = data.get('metadata', {}).get('row_id', i)
                tool_call_map[rid] = data.get('answer_key', [])
                full_map[rid] = data
    else:
        msg = "No answer key file found"
        if answer_key_path:
            msg += f" at {answer_key_path}"
        print(f"{msg} -- will rely on trajectory metadata only.")

    return tool_call_map, full_map


def write_jsonl(records, path):
    """Write a list of dicts as JSONL."""
    with open(path, 'w') as f:
        for item in records:
            f.write(json.dumps(item) + "\n")


def main():
    args = get_args()

    # Parse dimensions
    if args.dimensions.strip().lower() == "all":
        dimensions = ALL_DIMENSIONS
    else:
        dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        unknown = [d for d in dimensions if d not in DIMENSION_REGISTRY]
        if unknown:
            print(f"Error: unknown dimension(s): {unknown}")
            print(f"  Valid choices: {ALL_DIMENSIONS}")
            exit(1)

    print(f"Dimensions to prepare: {dimensions}")

    # Load data once
    trajectories = load_trajectories(args.input_file)
    tool_call_ak_map, full_ak_map = load_answer_keys(args.input_file, args.answer_key_file)

    base_name = args.input_file.replace(".jsonl", "")

    # Run each dimension
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"  Dimension: {dim}")
        print(f"{'='*60}")

        entry = DIMENSION_REGISTRY[dim]
        builder = entry["builder"]

        # tool_call uses the answer_key list map; others use the full record map
        if dim == "tool_call":
            ak_map = tool_call_ak_map
        else:
            ak_map = full_ak_map

        print(f"Processing {len(trajectories)} scenarios for [{dim}]...")
        eval_records, auto_records = builder(trajectories, ak_map)

        eval_file = f"{base_name}_eval_{dim}_prepared.jsonl"
        auto_file = f"{base_name}_eval_{dim}_auto_scores.jsonl"

        write_jsonl(eval_records, eval_file)
        write_jsonl(auto_records, auto_file)

        print(f"  - {len(eval_records)} eval prompts: {eval_file}")
        print(f"  - {len(auto_records)} auto-scored entries: {auto_file}")

    print(f"\nDone! Prepared {len(dimensions)} dimension(s).")


if __name__ == "__main__":
    main()
