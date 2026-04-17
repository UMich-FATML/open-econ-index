#!/usr/bin/env python3
"""
Map conversations from the WildChat dataset to O*NET occupational tasks
using embedding-based semantic search.

For each conversation summary marked as occupationally relevant, finds the
top-k closest O*NET tasks by cosine similarity using Qwen3-Embedding.

Outputs a JSON file mapping conversation_hash -> ([tasks], [summary]).
Supports resumable batch processing via a progress tracker file.
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from huggingface_hub import login


def get_all_tasks(tasks_file: str):
    """Load unique task statements from O*NET Task Statements file."""
    df = pd.read_csv(tasks_file, sep='\t')
    task_statements = df['Task'].tolist()
    print(f"Loaded {len(task_statements)} task statements")
    return task_statements


def create_mappings_file(IDs, summaries, task_names, task_embeddings,
                         embedding_model, k=3):
    """
    Embed conversation summaries and find top-k closest O*NET tasks by
    cosine similarity.

    Returns list of dicts with convo_id, summary, and matched tasks.
    """
    instruction_prompt = (
        "Embed this summary of a chat with an AI assistant "
        "to find a related occupational task: "
    )
    query_embeddings = embedding_model.encode(summaries, prompt=instruction_prompt)
    document_embeddings = task_embeddings

    similarity = embedding_model.similarity(query_embeddings, document_embeddings)
    similarity_np = similarity.cpu().numpy()
    top_k_indices = np.argsort(similarity_np, axis=1)[:, -k:][:, ::-1]

    task_mappings = []
    for i, (cid, summary) in enumerate(
        tqdm(zip(IDs, summaries), desc="Assembling tasks for summaries")
    ):
        top_indices = top_k_indices[i]
        tasks_for_summ = [task_names[idx] for idx in top_indices]
        task_mappings.append({
            "convo_id": cid,
            "summary": summary,
            "tasks": tasks_for_summ,
        })
    return task_mappings


def update_json_file(file_path, new_data):
    """Load existing JSON, merge new_data, and save back."""
    existing_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}. Starting fresh.")
    existing_data.update(new_data)
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Map WildChat conversations to O*NET tasks"
    )
    parser.add_argument(
        "--task_file", default="../onet_db/Task Statements.txt",
        help="Path to O*NET Task Statements TSV file",
    )
    parser.add_argument(
        "--source_dataset", default="umich-fatml/WildEnglishChats-1.6M",
        help="HuggingFace dataset with conversation summaries",
    )
    parser.add_argument(
        "--output_file", default="summaries2tasks.json",
        help="Output JSON mapping file",
    )
    parser.add_argument(
        "--embedding_model", default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence-transformer embedding model",
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--progress_file", default="progress_tracker.txt",
        help="File for resumable batch tracking",
    )
    args = parser.parse_args()

    # Authenticate with HuggingFace if token is set
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. Dataset loading may fail for gated datasets.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset {args.source_dataset}...")
    summaries_dataset = load_dataset(args.source_dataset, split="train", streaming=False)

    # Load tasks and encode them
    print(f"Loading tasks from {args.task_file}...")
    task_names = list(set(get_all_tasks(args.task_file)))
    embedding_model = SentenceTransformer(args.embedding_model)
    task_embeddings = embedding_model.encode(task_names)
    print(f"Task embeddings shape: {task_embeddings.shape}")

    # Filter to occupationally relevant summaries
    print("Filtering occupationally relevant summaries...")
    seen_ids = set()
    total_summaries = []
    total_ids = []
    for example in tqdm(summaries_dataset, desc="Filtering summaries"):
        convo_id = example["conversation_hash"]
        if convo_id not in seen_ids:
            if (len(example["relevance"]) > 0
                    and example["relevance"][0].lower() == "y"):
                total_summaries.append(example["request"])
                total_ids.append(convo_id)
                seen_ids.add(convo_id)

    # Resume from last checkpoint
    start_index = 0
    if os.path.exists(args.progress_file):
        with open(args.progress_file, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                start_index = int(content)
                print(f"Resuming from index {start_index}.")

    if start_index >= len(total_summaries):
        print("All conversations have already been processed.")
        return

    # Batch processing
    print(f"Processing {len(total_summaries)} summaries from index {start_index}...")
    for i in range(start_index, len(total_summaries), args.batch_size):
        end_index = min(i + args.batch_size, len(total_summaries))
        batch = total_summaries[i:end_index]
        batch_ids = total_ids[i:end_index]

        print(f"\n--- Batch: conversations {i} to {end_index - 1} ---")
        mappings_batch = create_mappings_file(
            IDs=batch_ids,
            summaries=batch,
            task_names=task_names,
            task_embeddings=task_embeddings,
            embedding_model=embedding_model,
            k=args.top_k,
        )

        mappings_data = {}
        for item in mappings_batch:
            mappings_data[item["convo_id"]] = (item["tasks"], [item["summary"]])

        update_json_file(args.output_file, mappings_data)

        with open(args.progress_file, 'w') as f:
            f.write(str(end_index))
        print(f"Progress saved. Next run starts from index {end_index}.")

    print("\nAll batches processed successfully!")


if __name__ == "__main__":
    main()
