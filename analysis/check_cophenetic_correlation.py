"""
Cophenetic correlation between Qwen3-Embedding distances and O*NET work activity hierarchy.

Measures how well embedding distances align with shortest path-length distances
in the GWA → IWA → DWA → Task hierarchy. Builds an undirected graph across all
hierarchy levels (GWA, IWA, DWA, Task), computes shortest-path distances using
scipy.sparse.csgraph.shortest_path, and cosine distances from embeddings, then
reports Pearson and Spearman correlations.
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

from utils import read_tsv, batch_encode_texts


def load_hierarchy(onet_dir: str):
    """
    Load the GWA → IWA → DWA → Task hierarchy from O*NET data files.

    Returns:
        gwas: dict {element_id: element_name}
        iwas: dict {iwa_id: {'title': str, 'gwa_id': str}}
        dwas: dict {dwa_id: {'title': str, 'iwa_id': str, 'gwa_id': str}}
        tasks: dict {task_id: {'text': str, 'dwa_ids': set}}
    """
    # Load GWAs from Work Activities.txt (unique Element ID + Element Name pairs)
    wa_rows = read_tsv(os.path.join(onet_dir, "Work Activities.txt"))
    gwas = {}
    for row in wa_rows:
        eid = row["Element ID"].strip()
        name = row["Element Name"].strip()
        if eid not in gwas:
            gwas[eid] = name
    print(f"Loaded {len(gwas)} GWAs")

    # Load IWAs from IWA Reference.txt
    iwa_rows = read_tsv(os.path.join(onet_dir, "IWA Reference.txt"))
    iwas = {}
    for row in iwa_rows:
        gwa_id = row["Element ID"].strip()
        iwa_id = row["IWA ID"].strip()
        title = row["IWA Title"].strip()
        iwas[iwa_id] = {"title": title, "gwa_id": gwa_id}
    print(f"Loaded {len(iwas)} IWAs")

    # Load DWAs from DWA Reference.txt
    dwa_rows = read_tsv(os.path.join(onet_dir, "DWA Reference.txt"))
    dwas = {}
    for row in dwa_rows:
        gwa_id = row["Element ID"].strip()
        iwa_id = row["IWA ID"].strip()
        dwa_id = row["DWA ID"].strip()
        title = row["DWA Title"].strip()
        dwas[dwa_id] = {"title": title, "iwa_id": iwa_id, "gwa_id": gwa_id}
    print(f"Loaded {len(dwas)} DWAs")

    # Load tasks from Task Statements.txt
    task_rows = read_tsv(os.path.join(onet_dir, "Task Statements.txt"))
    tasks = {}
    for row in task_rows:
        task_id = row["Task ID"].strip()
        text = row["Task"].strip()
        if task_id not in tasks:
            tasks[task_id] = {"text": text, "dwa_ids": set()}

    # Load Task→DWA mappings
    td_rows = read_tsv(os.path.join(onet_dir, "Tasks to DWAs.txt"))
    for row in td_rows:
        task_id = row["Task ID"].strip()
        dwa_id = row["DWA ID"].strip()
        if task_id in tasks:
            tasks[task_id]["dwa_ids"].add(dwa_id)

    # Filter to tasks that have at least one DWA mapping
    tasks = {k: v for k, v in tasks.items() if v["dwa_ids"]}
    print(f"Loaded {len(tasks)} tasks with DWA mappings")

    return gwas, iwas, dwas, tasks


def main():
    parser = argparse.ArgumentParser(
        description="Compute cophenetic correlation between embedding distances and O*NET hierarchy"
    )
    parser.add_argument(
        "--work-activity-embeds", type=str, default=None,
        help="Path to load/save cached activity embeddings (parquet). Default: {model}_work_activity_embeds.parquet"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model (default: Qwen/Qwen3-Embedding-8B)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for encoding (default: 16)"
    )
    parser.add_argument(
        "--max-length", type=int, default=8192,
        help="Maximum sequence length (default: 8192)"
    )
    parser.add_argument(
        "--onet-dir", type=str, default="../onet_db",
        help="Path to O*NET data directory"
    )
    args = parser.parse_args()

    # Load hierarchy
    gwas, iwas, dwas, tasks = load_hierarchy(args.onet_dir)

    # Collect all nodes across all levels
    node_ids, texts = [], []
    for gwa_id in sorted(gwas):
        node_ids.append(gwa_id)
        texts.append(gwas[gwa_id])
    for iwa_id in sorted(iwas):
        node_ids.append(iwa_id)
        texts.append(iwas[iwa_id]["title"])
    for dwa_id in sorted(dwas):
        node_ids.append(dwa_id)
        texts.append(dwas[dwa_id]["title"])
    for task_id in sorted(tasks):
        node_ids.append(task_id)
        texts.append(tasks[task_id]["text"])

    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    print(f"\nTotal nodes: {n}")
    print(f"  GWA: {len(gwas)}, IWA: {len(iwas)}, DWA: {len(dwas)}, Task: {len(tasks)}")

    # Build undirected graph as sparse adjacency matrix
    # Use (n+1) × (n+1) to include a virtual root node at index n
    # that connects to all GWAs, ensuring finite distances between all nodes
    print("\nBuilding hierarchy graph...")
    root_idx = n
    graph = lil_matrix((n + 1, n + 1), dtype=np.float32)

    # Root→GWA edges
    gwa_indices = [id_to_idx[gwa_id] for gwa_id in gwas]
    for gi in gwa_indices:
        graph[root_idx, gi] = graph[gi, root_idx] = 1

    # IWA→GWA edges
    for iwa_id, info in iwas.items():
        i, j = id_to_idx[iwa_id], id_to_idx[info["gwa_id"]]
        graph[i, j] = graph[j, i] = 1

    # DWA→IWA edges
    for dwa_id, info in dwas.items():
        i, j = id_to_idx[dwa_id], id_to_idx[info["iwa_id"]]
        graph[i, j] = graph[j, i] = 1

    # Task→DWA edges
    n_task_edges = 0
    for task_id, info in tasks.items():
        for dwa_id in info["dwa_ids"]:
            if dwa_id in id_to_idx:
                i, j = id_to_idx[task_id], id_to_idx[dwa_id]
                graph[i, j] = graph[j, i] = 1
                n_task_edges += 1

    print(f"  Root→GWA edges: {len(gwas)}")
    print(f"  IWA→GWA edges: {len(iwas)}")
    print(f"  DWA→IWA edges: {len(dwas)}")
    print(f"  Task→DWA edges: {n_task_edges}")

    # Compute shortest path distances on full (n+1) graph, then slice out root
    print("\nComputing shortest path distances...")
    dist_full = shortest_path(graph.tocsr(), directed=False, unweighted=True)
    dist_matrix = dist_full[:n, :n]  # exclude root node
    del dist_full
    dist_matrix[np.isinf(dist_matrix)] = np.iinfo(np.int8).max
    dist_matrix = dist_matrix.astype(np.int8)

    # Extract upper triangle as condensed vector
    triu_indices = np.triu_indices(n, k=1)
    coph_condensed = dist_matrix[triu_indices].astype(np.float32)

    # Report distance distribution
    unique_dists, dist_counts = np.unique(coph_condensed.astype(int), return_counts=True)
    print(f"\nShortest path distance distribution:")
    for d, c in zip(unique_dists, dist_counts):
        print(f"  distance={d}: {c:,} pairs")

    # Free the full matrix
    del dist_matrix

    # Get or compute embeddings
    model_short = args.model.split("/")[-1]
    embeds_path = args.work_activity_embeds or f"{model_short}_work_activity_embeds.parquet"
    embeddings = None
    if args.work_activity_embeds and os.path.exists(embeds_path):
        print(f"\nLoading cached embeddings from {embeds_path}")
        df = pd.read_parquet(embeds_path)
        cached_ids = df["ID"].tolist()
        if cached_ids == node_ids:
            embeddings = np.array(df["embedding"].tolist())
            print(f"Loaded {len(embeddings)} cached embeddings")
        else:
            print(f"WARNING: Cached embeddings don't match current nodes "
                  f"({len(cached_ids)} cached vs {len(node_ids)} expected). Re-encoding.")

    if embeddings is None:
        print(f"\nEncoding {len(texts)} texts with {args.model}...")
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to("cuda").eval()

        embeddings = batch_encode_texts(
            model, tokenizer, texts,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        # Always save embeddings cache
        df = pd.DataFrame({
            "ID": node_ids,
            "text": texts,
            "embedding": [emb.tolist() for emb in embeddings]
        })
        df.to_parquet(embeds_path, index=False)
        print(f"Saved embeddings to {embeds_path}")

    # Compute embedding cosine distances (condensed form)
    # Since embeddings are L2-normalized, cosine distance = 1 - dot product
    print("\nComputing embedding distances...")
    emb_condensed = np.empty(n * (n - 1) // 2, dtype=np.float32)
    # Process in blocks to avoid materializing full n×n matrix
    block_size = 500
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        for j in range(i, n, block_size):
            end_j = min(j + block_size, n)
            # Compute dot products for this block
            dots = embeddings[i:end_i] @ embeddings[j:end_j].T
            # Extract upper-triangle entries that belong to the condensed vector
            for bi in range(end_i - i):
                gi = i + bi  # global row index
                # For this row, we need columns > gi
                start_col = max(gi + 1, j)
                if start_col >= end_j:
                    continue
                bj_start = start_col - j
                bj_end = end_j - j
                # Position in condensed vector for pair (gi, start_col)
                pos = gi * n - gi * (gi + 1) // 2 + (start_col - gi - 1)
                count = bj_end - bj_start
                emb_condensed[pos:pos+count] = 1.0 - dots[bi, bj_start:bj_end]

    # Compute correlations
    print("\nComputing correlations...")
    pearson_r, pearson_p = pearsonr(coph_condensed, emb_condensed)
    spearman_r, spearman_p = spearmanr(coph_condensed, emb_condensed)

    # Mean embedding distance per shortest path distance
    mean_emb_by_coph = {}
    for d in unique_dists:
        mask = coph_condensed == d
        mean_emb_by_coph[int(d)] = {
            "mean_cosine_distance": float(np.mean(emb_condensed[mask])),
            "std_cosine_distance": float(np.std(emb_condensed[mask])),
            "count": int(np.sum(mask))
        }

    # Report
    print("\n" + "=" * 60)
    print("Cophenetic Correlation Analysis (all levels)")
    print(f"Nodes: {n} (GWA={len(gwas)}, IWA={len(iwas)}, DWA={len(dwas)}, Task={len(tasks)})")
    print(f"Pairs: {len(coph_condensed):,}")
    print("=" * 60)
    print(f"\nPearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")
    print(f"\nMean cosine distance by shortest path distance:")
    for d in unique_dists:
        info = mean_emb_by_coph[int(d)]
        print(f"  path={d}: mean={info['mean_cosine_distance']:.4f} "
              f"± {info['std_cosine_distance']:.4f}  (n={info['count']:,})")


if __name__ == "__main__":
    main()
