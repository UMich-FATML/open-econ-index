"""
Match O*NET software commodities to Smithery MCP servers as whole units.

Uses Qwen3-Embedding for semantic matching between commodity definitions
and Smithery MCP server documents (combining server metadata and all tool info).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from utils import (
    save_dataset, read_tsv, load_commodities, format_commodity_as_query,
    batch_encode_texts,
)


def format_server_as_document(
  json_path: Path,
  include_tool_inputs: bool = False,
  validated_only: bool = False
) -> Optional[Tuple[Dict, str]]:
  """
  Load a Smithery MCP server JSON file and format its metadata as document text.

  Args:
    json_path: Path to the MCP server JSON file
    include_tool_inputs: Whether to include tool input parameter names and descriptions

  Returns:
    Tuple of (server_metadata_dict, document_text) or None if server is invalid
  """
  try:
    with open(json_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
  except Exception as e:
    print(f"Warning: Failed to load {json_path}: {e}")
    return None

  server = data.get('server') or {}

  # Skip servers that failed validation
  if validated_only and server.get('validation_error') is not None:
    return None

  # Skip servers with no tools
  server_tools = server.get('tools') or []
  if not server_tools:
    return None

  # Skip servers with no analysis
  analysis = data.get('analysis', '')
  if not analysis:
    return None

  server_name = server.get('displayName', '')
  description = server.get('description', '')
  labels = data.get('labels') or []
  categories = data.get('categories') or []

  # Build document text
  parts = []
  if server_name:
    parts.append(f"Server: {server_name}")
  if description:
    parts.append(f"Description: {description}")
  if analysis:
    parts.append(f"Analysis: {analysis}")
  if labels:
    parts.append(f"Categories: {', '.join(labels)}")
  if categories:
    parts.append(f"Tags: {', '.join(categories)}")

  # Add tools
  tool_lines = []
  for tool in server_tools:
    tool_name = tool.get('name', '')
    tool_desc = tool.get('description', '')
    if tool_name and tool_desc:
      tool_line = f"- {tool_name}: {tool_desc}"
    elif tool_name:
      tool_line = f"- {tool_name}"
    else:
      continue

    if include_tool_inputs:
      input_schema = tool.get('inputSchema', tool.get('input_schema', {}))
      properties = input_schema.get('properties', {}) if isinstance(input_schema, dict) else {}
      if properties:
        param_lines = []
        for param_name, param_info in properties.items():
          if not isinstance(param_info, dict):
            param_lines.append(f"    - {param_name}")
            continue
          param_desc = param_info.get('description', '')
          if param_desc:
            param_lines.append(f"    - {param_name}: {param_desc}")
          else:
            param_lines.append(f"    - {param_name}")
        if param_lines:
          tool_line += "\n    Parameters:\n" + "\n".join(param_lines)

    tool_lines.append(tool_line)

  if tool_lines:
    parts.append("Tools:\n" + "\n".join(tool_lines))

  document_text = "\n".join(parts)

  server_meta = data  # full JSON file contents
  server_meta['filename'] = json_path.name  # add filename since it's not in the JSON

  return server_meta, document_text


def save_server_embeds_parquet(
  servers: List[Dict],
  document_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save server embeddings with lightweight cache data to parquet file."""
  df = pd.DataFrame({
    'server_name': [s.get('server', {}).get('displayName', '') for s in servers],
    'filename': [s['filename'] for s in servers],
    'document_text': document_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved server embeddings to {filepath}")


def load_server_embeds_parquet(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
  """Load server embeddings from parquet file.

  Returns:
    Tuple of (filenames, document_texts, embeddings)
  """
  print(f"Loading server embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  filenames = df['filename'].tolist()
  document_texts = df['document_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(filenames)} server embeddings")
  return filenames, document_texts, embeddings


def save_commodity_embeds_parquet(
  commodities_df: pd.DataFrame,
  query_texts: List[str],
  commodity_indices: List[int],
  embeddings: np.ndarray,
  filepath: str
):
  """Save multi-query commodity embeddings to parquet file.

  Each row is one query (one example for a commodity). Multiple rows can
  share the same commodity_idx.
  """
  df = pd.DataFrame({
    'commodity_idx': commodity_indices,
    'commodity_code': [commodities_df.iloc[i]['code'] for i in commodity_indices],
    'commodity_title': [commodities_df.iloc[i]['title'] for i in commodity_indices],
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved {len(df)} commodity query embeddings to {filepath}")


def load_commodity_embeds_parquet(
  filepath: str
) -> Tuple[List[str], List[int], np.ndarray]:
  """Load commodity embeddings from parquet file.

  Handles both multi-query format (with commodity_idx column) and
  legacy single-query format (one row per commodity, no commodity_idx).

  Returns:
    Tuple of (query_texts, commodity_indices, embeddings)
  """
  print(f"Loading commodity embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  query_texts = df['query_text'].tolist()
  if 'commodity_idx' in df.columns:
    commodity_indices = df['commodity_idx'].tolist()
  else:
    # Legacy format: one row per commodity, sequential indices
    commodity_indices = list(range(len(df)))
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(df)} commodity query embeddings")
  return query_texts, commodity_indices, embeddings


def compute_commodity_server_coverage(
  commodity_embeds: np.ndarray,
  server_embeds: np.ndarray,
  commodities_df: pd.DataFrame,
  servers: List[Dict],
  commodity_indices: List[int],
  threshold: float = 0.5
) -> Tuple[List[Dict], Dict]:
  """
  Match commodities to servers and compute coverage statistics.

  When a commodity has multiple query embeddings (one per example),
  takes the max similarity across queries for each server (union semantics).

  Args:
    commodity_embeds: Shape (num_queries, hidden_dim), may have multiple rows per commodity
    server_embeds: Shape (num_servers, hidden_dim)
    commodities_df: DataFrame with code, title, definition, examples
    servers: List of server metadata dicts
    commodity_indices: Maps each query row to its commodity index in commodities_df
    threshold: Minimum similarity score to include

  Returns:
    Tuple of (results list, stats dict)
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Cosine similarity via dot product (embeddings are normalized)
  similarities = np.dot(commodity_embeds, server_embeds.T)  # (num_queries, num_servers)

  # Build lookup: commodity index -> list of query row indices
  commodity_to_query_rows: Dict[int, List[int]] = {}
  for query_row, comm_idx in enumerate(commodity_indices):
    commodity_to_query_rows.setdefault(comm_idx, []).append(query_row)

  results = []
  covered_commodities = set()
  matched_server_idxs = set()

  for i, row in tqdm(commodities_df.iterrows(), desc="Matching commodities", total=len(commodities_df)):
    query_rows = commodity_to_query_rows.get(i, [])

    # Max similarity across all queries for this commodity (union semantics)
    if len(query_rows) == 1:
      sims = similarities[query_rows[0]]
    else:
      sims = similarities[query_rows].max(axis=0)

    # Find all servers above threshold
    above_threshold = np.where(sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(sims[above_threshold])[::-1]]

    matched_servers = []
    for idx in sorted_indices:
      server = servers[idx]
      match = dict(server)  # shallow copy of full JSON contents
      match['server_idx'] = int(idx)
      match['similarity_score'] = float(sims[idx])
      matched_servers.append(match)
      matched_server_idxs.add(idx)

    if matched_servers:
      covered_commodities.add(row['title'])

    results.append({
      'commodity_code': row['code'],
      'commodity_title': row['title'],
      'commodity_definition': row['definition'],
      'matched_servers': matched_servers
    })

  # Compute statistics
  all_titles = commodities_df['title'].tolist()
  uncovered = [t for t in all_titles if t not in covered_commodities]
  stats = {
    'total_commodities': len(commodities_df),
    'covered_commodities': len(covered_commodities),
    'coverage_rate': len(covered_commodities) / len(commodities_df) if len(commodities_df) else 0,
    'total_servers': len(servers),
    'matched_servers': len(matched_server_idxs),
    'utilization_rate': len(matched_server_idxs) / len(servers) if servers else 0,
    'uncovered_commodities': uncovered
  }

  return results, stats


def print_coverage_report(stats: Dict):
  """Print coverage summary statistics to stdout."""
  print(f"\n{'='*60}")
  print("Coverage Summary")
  print(f"{'='*60}")
  print(f"Commodities: {stats['covered_commodities']} / {stats['total_commodities']} covered "
        f"({stats['coverage_rate']:.1%})")
  print(f"Servers: {stats['matched_servers']} / {stats['total_servers']} matched "
        f"({stats['utilization_rate']:.1%})")

  if stats['uncovered_commodities']:
    print(f"\n{'='*60}")
    print(f"Uncovered Commodity Titles ({len(stats['uncovered_commodities'])})")
    print(f"{'='*60}")
    for title in sorted(stats['uncovered_commodities']):
      print(f"  - {title}")


def main():
  parser = argparse.ArgumentParser(
    description='Match O*NET commodities to Smithery MCP servers using Qwen3-Embedding'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers/smithery_mcp_servers_0210',
    help='Path to directory containing Smithery MCP server JSON files'
  )
  parser.add_argument(
    '--server-embeds',
    default=None,
    help='Server embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--commodity-embeds',
    default=None,
    help='Commodity embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--tech-skills-file',
    default='../onet_db/Technology Skills.txt',
    help='Path to O*NET Technology Skills TSV file'
  )
  parser.add_argument(
    '--unspsc-file',
    default='../onet_db/unspsc-english-v260801.1.csv',
    help='Path to UNSPSC CSV file with commodity definitions'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--include-commodity-examples',
    action='store_true',
    help='Include commodity software examples in query text'
  )
  parser.add_argument(
    '--include-tool-inputs',
    action='store_true',
    help='Include tool input parameter names and descriptions in server document text'
  )
  parser.add_argument(
    '--validated-only',
    action='store_true',
    help='Only include servers with no validation_error (server.validation_error is null)'
  )
  parser.add_argument(
    '--output-file',
    default='commodities_to_smithery_servers.jsonl',
    help='Output JSONL file path for detailed results'
  )
  parser.add_argument(
    '--model',
    default='Qwen/Qwen3-Embedding-8B',
    help='Qwen3-Embedding model to use'
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='Batch size for encoding'
  )
  parser.add_argument(
    '--max-length',
    type=int,
    default=8192,
    help='Maximum sequence length for tokenization'
  )

  args = parser.parse_args()

  # Default embedding file paths
  model_short = args.model.split('/')[-1]
  commodity_embeds_file = args.commodity_embeds or f"{model_short}_commodity_embeds.parquet"
  server_embeds_file = args.server_embeds or f"{model_short}_smithery_server_embeds.parquet"

  # Load commodities
  print(f"Loading commodity data from {args.tech_skills_file}...")
  commodities_df = load_commodities(args.tech_skills_file, args.unspsc_file)

  # Load MCP servers
  mcp_path = Path(args.mcp_dir)
  json_files = sorted(mcp_path.glob("*.json"))
  print(f"Loading MCP servers from {args.mcp_dir}...")
  servers = []
  document_texts = []
  for json_file in tqdm(json_files, desc="Loading MCP servers"):
    result = format_server_as_document(json_file, include_tool_inputs=args.include_tool_inputs, validated_only=args.validated_only)
    if result is not None:
      server_meta, doc_text = result
      servers.append(server_meta)
      document_texts.append(doc_text)
  print(f"Loaded {len(servers)} valid MCP servers")

  # Check if we can load cached embeddings
  load_commodity_cache = args.commodity_embeds and os.path.exists(args.commodity_embeds)
  load_server_cache = args.server_embeds and os.path.exists(args.server_embeds)
  need_model = not (load_commodity_cache and load_server_cache)

  # Model and tokenizer (only load if needed)
  model = None
  tokenizer = None
  device = None

  if need_model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    if device == "cuda" and torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

  # Load or compute commodity embeddings
  if load_commodity_cache:
    cached_titles, query_texts, commodity_indices, commodity_embeds = None, None, None, None
    try:
      query_texts, commodity_indices, commodity_embeds = load_commodity_embeds_parquet(args.commodity_embeds)
      # Read back commodity titles for consistency check
      cached_df = pd.read_parquet(args.commodity_embeds)
      cached_titles = cached_df['commodity_title'].unique().tolist()
    except Exception as e:
      print(f"Warning: Failed to load commodity cache: {e}. Recomputing...")
      load_commodity_cache = False

    if load_commodity_cache and sorted(cached_titles) != sorted(commodities_df['title'].tolist()):
      print("Warning: Cached commodity titles differ from current file. Recomputing...")
      load_commodity_cache = False

  if not load_commodity_cache:
    print("Building query texts for commodities...")
    query_texts = []
    commodity_indices = []
    for i, row in commodities_df.iterrows():
      if args.include_commodity_examples and row['examples']:
        for ex in row['examples']:
          query_texts.append(format_commodity_as_query(row['definition'], examples=[ex]))
          commodity_indices.append(i)
      else:
        query_texts.append(format_commodity_as_query(row['definition']))
        commodity_indices.append(i)

    print(f"\nEncoding {len(query_texts)} commodity queries ({len(commodities_df)} commodities)...")
    commodity_embeds = batch_encode_texts(
      model, tokenizer, query_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_commodity_embeds_parquet(
      commodities_df, query_texts, commodity_indices, commodity_embeds, commodity_embeds_file
    )

  # Load or compute server embeddings
  if load_server_cache:
    cached_filenames, cached_doc_texts, server_embeds = load_server_embeds_parquet(args.server_embeds)
    current_filenames = [s['filename'] for s in servers]
    if cached_filenames != current_filenames:
      print("Warning: Cached server filenames differ from current. Recomputing...")
      load_server_cache = False
    else:
      document_texts = cached_doc_texts
      # servers list stays as-is (full JSON data from loading step)

  if not load_server_cache:
    print(f"\nEncoding {len(document_texts)} MCP server documents...")
    server_embeds = batch_encode_texts(
      model, tokenizer, document_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_server_embeds_parquet(servers, document_texts, server_embeds, server_embeds_file)

  # Verify embeddings are normalized
  print("\nVerifying embeddings are normalized...")
  commodity_norms = np.linalg.norm(commodity_embeds, axis=1)
  server_norms = np.linalg.norm(server_embeds, axis=1)
  print(f"Commodity embedding norms - mean: {commodity_norms.mean():.4f}, std: {commodity_norms.std():.6f}")
  print(f"Server embedding norms - mean: {server_norms.mean():.4f}, std: {server_norms.std():.6f}")

  # Compute coverage
  results, stats = compute_commodity_server_coverage(
    commodity_embeds, server_embeds,
    commodities_df, servers,
    commodity_indices=commodity_indices,
    threshold=args.threshold
  )

  # Print coverage report
  print_coverage_report(stats)

  # Save detailed results
  print(f"\nSaving results to {args.output_file}...")
  save_dataset(results, args.output_file, convert_to_jsonl=True)
  print(f"Done! Saved {len(results)} commodity results to {args.output_file}")


if __name__ == "__main__":
  main()
