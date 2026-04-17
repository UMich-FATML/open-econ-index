import os
import csv
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

################
# File I/O
################

# Save dataset
def save_dataset(data, filename, convert_to_jsonl=False):
  if convert_to_jsonl:
    with open(filename, 'w') as file:
      for obj in data:
        file.write(json.dumps(obj) + '\n')
  else:
    with open(filename, 'w') as file:
      json.dump(data, file, indent=2)


################
# Data loading
################

def read_tsv(filepath: str) -> List[Dict]:
  """Read a tab-separated file and return list of dicts."""
  with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    return list(reader)


def load_commodities(
  tech_skills_file: str,
  unspsc_file: str = None
) -> pd.DataFrame:
  """
  Load unique commodities with codes, titles, definitions, and example software.

  Args:
    tech_skills_file: Path to Technology Skills.txt
    unspsc_file: Optional path to UNSPSC CSV file with commodity definitions

  Returns:
    DataFrame with columns: code, title, definition, examples
  """
  # Load definitions from UNSPSC file if provided
  definitions: Dict[str, str] = {}
  if unspsc_file and os.path.exists(unspsc_file):
    with open(unspsc_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
        code = row.get('Commodity', '').strip()
        definition = row.get('Commodity Definition', '').strip()
        if code and definition:
          definitions[code] = definition
    print(f"Loaded {len(definitions)} commodity definitions from UNSPSC file")

  # Load commodity data from Technology Skills file
  raw_data = read_tsv(tech_skills_file)

  # Collect codes and examples for each commodity title
  commodity_codes: Dict[str, str] = {}
  commodity_examples: Dict[str, set] = {}
  for row in raw_data:
    title = row.get('Commodity Title', '').strip()
    code = row.get('Commodity Code', '').strip()
    example = row.get('Example', '').strip()
    if title and code:
      commodity_codes[title] = code
      if title not in commodity_examples:
        commodity_examples[title] = set()
      if example:
        commodity_examples[title].add(example)

  # Build DataFrame sorted by title
  unique_titles = sorted(commodity_codes.keys())
  rows = []
  for title in unique_titles:
    code = commodity_codes[title]
    rows.append({
      'code': code,
      'title': title,
      'definition': definitions.get(code, ''),
      'examples': sorted(commodity_examples.get(title, set()))
    })

  df = pd.DataFrame(rows)

  if definitions:
    num_with_defs = sum(1 for d in df['definition'] if d)
    print(f"Matched {num_with_defs}/{len(df)} commodities with definitions")

  print(f"Loaded {len(df)} unique commodities")
  return df


def format_commodity_as_query(
  definition: str,
  examples: Optional[List[str]] = None,
) -> str:
  """
  Format a commodity as an instruction-aware query for Qwen3-Embedding.

  Args:
    definition: Commodity definition from UNSPSC
    examples: Optional list of example software names

  Returns:
    Instruction-prefixed query string
  """
  query = definition
  if examples:
    query = f"{definition} such as {', '.join(examples)}."
  return (
    f"Instruct: Retrieve software tools that provide similar functionality as software in the following category.\n"
    f"Query: {query}"
  )


################
# Embedding
################

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  """
  Extract the last token's hidden state for each sequence in the batch.
  This is the convention for Qwen3-Embedding models.

  Args:
    last_hidden_states: Shape (batch_size, seq_len, hidden_dim)
    attention_mask: Shape (batch_size, seq_len)

  Returns:
    Pooled embeddings of shape (batch_size, hidden_dim)
  """
  left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
  if left_padding:
    return last_hidden_states[:, -1]
  else:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def batch_encode_texts(
  model: AutoModel,
  tokenizer: AutoTokenizer,
  texts: List[str],
  batch_size: int = 8,
  max_length: int = 8192,
  device: str = "cuda"
) -> np.ndarray:
  """
  Batch encode texts using Qwen3-Embedding with last_token_pool.

  Args:
    model: The Qwen3-Embedding model
    tokenizer: The tokenizer
    texts: List of texts to encode
    batch_size: Batch size for encoding
    max_length: Maximum sequence length
    device: Device to use

  Returns:
    Normalized embeddings of shape (num_texts, hidden_dim)
  """
  embeds = []

  for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
    batch_texts = texts[i:i + batch_size]

    # Tokenize
    batch_dict = tokenizer(
      batch_texts,
      max_length=max_length,
      padding=True,
      truncation=True,
      return_tensors="pt"
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    # Forward pass
    with torch.no_grad():
      outputs = model(**batch_dict)
      _embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize
    _embeds = torch.nn.functional.normalize(_embeds, p=2, dim=1)
    embeds.append(_embeds.cpu().numpy())

  return np.vstack(embeds)


################
# Embedding cache (commodities)
################

def save_commodity_embeds_parquet(
  commodities_df: pd.DataFrame,
  query_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save commodity embeddings with codes, definitions, and texts to parquet file."""
  df = pd.DataFrame({
    'commodity_code': commodities_df['code'].tolist(),
    'commodity_title': commodities_df['title'].tolist(),
    'commodity_definition': commodities_df['definition'].tolist(),
    'commodity_examples': [json.dumps(ex) for ex in commodities_df['examples'].tolist()],
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved commodity embeddings to {filepath}")


def load_commodity_embeds_parquet(filepath: str) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
  """Load commodity embeddings from parquet file."""
  print(f"Loading commodity embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  commodities_df = pd.DataFrame({
    'code': df['commodity_code'],
    'title': df['commodity_title'],
    'definition': df['commodity_definition'] if 'commodity_definition' in df.columns else [''] * len(df),
    'examples': [json.loads(ex) if ex else [] for ex in df.get('commodity_examples', ['[]'] * len(df))]
  })
  query_texts = df['query_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  return commodities_df, query_texts, embeddings
