#!/usr/bin/env python3
"""
prune_vocab.py — Remove specified vocabulary token IDs from a Hugging Face
causal language model and write out a pruned model.

The tokenizer is rebuilt by directly manipulating the tokenizer.json vocabulary,
preserving the original tokenization algorithm, normalizer, pre-tokenizer,
decoder, post-processor, and added tokens.  The pruned model can be loaded
directly with ``AutoModelForCausalLM.from_pretrained()`` and
``AutoTokenizer.from_pretrained()``.

Requirements
------------
- Python ≥ 3.11
- torch
- transformers
- safetensors

Usage
-----
python prune_vocab.py --model <input_dir> --output <output_dir> \\
                       --remove-ids <comma-separated IDs> [--dry-run]

Inspect tensors::

    python prune_vocab.py --model <input_dir> --explain-tensors
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Set

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove specified vocabulary token IDs from a Hugging Face "
            "causal language model and write out a pruned model."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the input Hugging Face model directory.",
    )
    parser.add_argument(
        "--output",
        help="Path where the pruned model will be saved.",
    )
    parser.add_argument(
        "--remove-ids",
        default="",
        help="Comma-separated list of token IDs to remove (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--remove-ids-file",
        default="",
        help="Path to a text file with one token ID per line to remove.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report what would change without writing files.",
    )
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help="Pass ``ignore_mismatched_sizes=True`` to ``from_pretrained``. "
        "Use when loading quantized or custom models whose weight shapes "
        "differ from the architecture config.",
    )
    parser.add_argument(
        "--explain-tensors",
        action="store_true",
        help="Print detailed explanations of every tensor in the checkpoint.\n"
        "Requires --model; --output and --remove-ids are ignored.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Token ID helpers  (unit-testable)
# ---------------------------------------------------------------------------


def parse_remove_ids(ids_str: str) -> Set[int]:
    """Convert a comma-separated string of integers into a set of ints.

    Parameters
    ----------
    ids_str : str
        e.g. ``"0,1,2"``

    Returns
    -------
    Set[int]
        Parsed token IDs.

    Raises
    ------
    SystemExit
        If any token is not a valid integer.
    """
    ids: Set[int] = set()
    for part in ids_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            print(f"Error: '{part}' is not a valid integer token ID.")
            sys.exit(1)
    return ids


def validate_token_ids(remove_ids: Set[int], vocab_size: int) -> None:
    """Exit with an error if any ID in *remove_ids* is outside [0, vocab_size).

    Parameters
    ----------
    remove_ids : Set[int]
        Token IDs to validate.
    vocab_size : int
        Size of the original vocabulary.

    Raises
    ------
    SystemExit
        If any ID is out of range.
    """
    for tid in sorted(remove_ids):
        if not (0 <= tid < vocab_size):
            print(
                f"Error: token ID {tid} is out of range. "
                f"Vocabulary size is {vocab_size} "
                f"(valid range: 0\u2013{vocab_size - 1})."
            )
            sys.exit(1)


def build_kept_ids(remove_ids: Set[int], vocab_size: int) -> List[int]:
    """Return a sorted list of token IDs that are *not* in *remove_ids*.

    Parameters
    ----------
    remove_ids : Set[int]
        Token IDs to exclude.
    vocab_size : int
        Total vocabulary size.

    Returns
    -------
    List[int]
        Sorted kept token IDs.
    """
    return [i for i in range(vocab_size) if i not in remove_ids]


def build_id_mapping(kept_ids: List[int]) -> Dict[int, int]:
    """Map old token IDs to new consecutive IDs starting from 0.

    ``result[old_id] == new_id`` for every old_id in *kept_ids*.

    Parameters
    ----------
    kept_ids : List[int]
        Sorted list of token IDs to keep.

    Returns
    -------
    Dict[int, int]
        old_id → new_id mapping.
    """
    return {old_id: new_id for new_id, old_id in enumerate(kept_ids)}


# ---------------------------------------------------------------------------
# Tensor pruning  (unit-testable)
# ---------------------------------------------------------------------------


def prune_weight(
    weight: torch.Tensor,
    id_mapping: Dict[int, int],
    new_vocab_size: int,
) -> torch.Tensor:
    """Select and reorder rows of a 2-D weight tensor according to *id_mapping*.

    ``result[new_id] = weight[old_id]`` for each ``old_id → new_id`` entry.

    Parameters
    ----------
    weight : torch.Tensor
        Original 2-D weight tensor (vocab_size, hidden_dim).
    id_mapping : Dict[int, int]
        old_id → new_id mapping.
    new_vocab_size : int
        Size of the pruned vocabulary.

    Returns
    -------
    torch.Tensor
        Pruned weight tensor of shape (new_vocab_size, hidden_dim).
    """
    assert weight.dim() == 2, f"Expected a 2-D weight tensor, got shape {weight.shape}"
    old_indices = torch.zeros(new_vocab_size, dtype=torch.long, device=weight.device)
    for old_id, new_id in id_mapping.items():
        old_indices[new_id] = old_id
    return weight.index_select(0, old_indices)


def prune_embedding_weight(
    embed: torch.nn.Embedding,
    id_mapping: Dict[int, int],
    new_vocab_size: int,
) -> torch.Tensor:
    """Return a pruned weight tensor for an embedding layer.

    Parameters
    ----------
    embed : torch.nn.Embedding
        The token embedding layer.
    id_mapping : Dict[int, int]
        old_id → new_id mapping.
    new_vocab_size : int
        Target vocabulary size.

    Returns
    -------
    torch.Tensor
        Pruned embedding weight.
    """
    return prune_weight(embed.weight.data, id_mapping, new_vocab_size)


def prune_lm_head_weight(
    linear: torch.nn.Linear,
    id_mapping: Dict[int, int],
    new_vocab_size: int,
) -> torch.Tensor:
    """Return a pruned weight tensor for the LM head layer.

    Parameters
    ----------
    linear : torch.nn.Linear
        The LM head linear layer.
    id_mapping : Dict[int, int]
        old_id → new_id mapping.
    new_vocab_size : int
        Target vocabulary size.

    Returns
    -------
    torch.Tensor
        Pruned LM head weight.
    """
    return prune_weight(linear.weight.data, id_mapping, new_vocab_size)


# ---------------------------------------------------------------------------
# Safetensors vocabulary pruning  (unit-testable)
# ---------------------------------------------------------------------------

# Tensor name prefixes that indicate vocabulary-related weights.
_VOCAB_PREFIXES = [
    "model.embed_tokens",
    "model.model.embed_tokens",
    "transformer.wte",
    "gpt_neox.embed_in",
    "embed",
    "lm_head",
]


def is_vocab_tensor(name: str, shape: torch.Size, vocab_size: int) -> bool:
    """Return ``True`` if *name*/*shape* refer to a vocabulary-mapped tensor.

    A tensor is considered vocab-mapped when:
    1. Its first dimension equals *vocab_size*, **and**
    2. Its name matches a known embedding/lm-head pattern.

    Matching uses both prefix-based (e.g. ``model.embed_tokens``) and
    substring-based (e.g. ``embed_tokens.weight``) checks to accommodate
    different model architectures (Qwen, Llama, GPT-2, etc.).
    """
    if len(shape) < 1 or shape[0] != vocab_size:
        return False
    lower = name.lower()
    # Fast path: known prefix matches
    if any(lower.startswith(p.lower()) for p in _VOCAB_PREFIXES):
        return True
    # Flexible match: any tensor ending with embed_tokens.weight
    # or starting with lm_head is vocab-mapped.
    # The shape[0] == vocab_size guard prevents false positives.
    if lower.endswith("embed_tokens.weight"):
        return True
    return False


def find_vocab_tensors(
    tensors: dict,
    vocab_size: int,
) -> dict:
    """Return ``{name: tensor}`` for every vocab-mapped tensor in *tensors*."""
    return {
        name: tensor
        for name, tensor in tensors.items()
        if is_vocab_tensor(name, tensor.size(), vocab_size)
    }


def prune_safetensors_directory(
    input_dir: str,
    output_dir: str,
    remove_ids: Set[int],
    vocab_size: int,
    id_mapping: Dict[int, int],
    new_vocab_size: int,
) -> None:
    """Prune vocabulary from every ``.safetensors`` file in *input_dir*.

    For each shard:
    1. Load tensors with ``safetensors.torch.load_file``.
    2. Find any vocab-mapped tensors (embedding, lm-head, and their
       companion quantisation scale tensors).
    3. Prune the first dimension of those tensors using the *id_mapping*.
    4. Save the modified shard to *output_dir*.

    Files that contain no vocab-mapped tensors are copied verbatim so that
    quantisation metadata and other auxiliary weights are preserved
    exactly.
    """
    safetensors_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        print("Error: no .safetensors files found in model directory.")
        sys.exit(1)

    for fname in safetensors_files:
        src = os.path.join(input_dir, fname)
        tensors = load_file(src, device="cpu")

        vocab_tensors = find_vocab_tensors(tensors, vocab_size)
        if not vocab_tensors:
            # No vocab-related tensors — plain copy
            shutil.copy2(src, os.path.join(output_dir, fname))
            continue

        for name, tensor in vocab_tensors.items():
            tensors[name] = prune_weight(tensor, id_mapping, new_vocab_size)

        dst = os.path.join(output_dir, fname)
        save_file(tensors, dst)

    # Copy safetensors index if it exists (maps tensor names to shard files)
    for idx_name in ("model.safetensors.index.json",):
        src_idx = os.path.join(input_dir, idx_name)
        if os.path.exists(src_idx):
            shutil.copy2(src_idx, os.path.join(output_dir, idx_name))


# ---------------------------------------------------------------------------
# Tokenizer rebuilding  (unit-testable helpers)
# ---------------------------------------------------------------------------


def detect_tokenizer_type(tokenizer) -> str:
    """Return a human-readable string describing the tokenizer's backend.

    Inspects the ``backend_tokenizer.model`` class name, or falls back
    to heuristics for legacy tokenizers.
    """
    if hasattr(tokenizer, "backend_tokenizer") and tokenizer.backend_tokenizer is not None:
        return type(tokenizer.backend_tokenizer.model).__name__
    if hasattr(tokenizer, "sp_model"):
        return "SentencePiece"
    if hasattr(tokenizer, "vocab"):
        return "BPE" if hasattr(tokenizer, "merges") else "WordPiece"
    return "Unknown"


def _rebuild_added_tokens(
    added_tokens: List[dict],
    remove_ids: Set[int],
    id_mapping: Dict[int, int],
) -> List[dict]:
    """Remove added-token entries whose IDs are in *remove_ids* and renumber."""
    result: List[dict] = []
    for entry in added_tokens:
        old_id = entry.get("id")
        if old_id is not None and old_id not in remove_ids:
            entry = dict(entry)
            entry["id"] = id_mapping[old_id]
            result.append(entry)
    return result


def rebuild_tokenizer_json(
    tokenizer_data: dict,
    remove_ids: Set[int],
    id_mapping: Dict[int, int],
) -> dict:
    """Return a modified copy of a parsed ``tokenizer.json`` dict.

    Handles the following vocabulary representations:

    * **BPE / WordPiece** – ``model.vocab`` is a ``{token: id}`` dict
    * **Unigram / SentencePiece** – ``model.vocab`` is a ``[[token, score], ...]``
      list where the index is the token ID

    All other tokenizer components (normalizer, pre_tokenizer, decoder,
    post_processor, …) are preserved unchanged.
    """
    data = json.loads(json.dumps(tokenizer_data))
    model = data.get("model", {})
    model_type = model.get("type", "")

    if model_type in ("BPE", "WordPiece"):
        # Dict-based vocabulary: {token_string: id}
        old_vocab: Dict[str, int] = model.get("vocab", {})
        new_vocab: Dict[str, int] = {}
        for token, old_id in old_vocab.items():
            if old_id not in remove_ids:
                new_vocab[token] = id_mapping[old_id]
        model["vocab"] = new_vocab

        # For BPE, filter merges that reference tokens no longer in the vocab
        # or whose result token was removed.
        if model_type == "BPE" and "merges" in model:
            new_vocab_set = set(new_vocab.keys())
            filtered_merges = []
            for merge_str in model["merges"]:
                parts = merge_str.split(" ", 1)
                if len(parts) == 2:
                    t1, t2 = parts
                    result = t1 + t2
                    if t1 in new_vocab_set and t2 in new_vocab_set and result in new_vocab_set:
                        filtered_merges.append(merge_str)
            model["merges"] = filtered_merges

    elif model_type in ("Unigram", "SentencePiece"):
        # List-based vocabulary: [[token, score], ...] where index = ID
        old_vocab_list: List = model.get("vocab", [])
        new_vocab_list: List = []
        for old_id, entry in enumerate(old_vocab_list):
            if old_id not in remove_ids:
                new_vocab_list.append(entry)
        model["vocab"] = new_vocab_list

    else:
        # Fallback: attempt dict-style pruning with a warning
        print(
            f"Warning: unknown tokenizer model type '{model_type}'. "
            "Attempting dict-style vocabulary pruning."
        )
        if isinstance(model.get("vocab"), dict):
            old_vocab_fb: Dict[str, int] = model["vocab"]
            new_vocab_fb: Dict[str, int] = {}
            for token, old_id in old_vocab_fb.items():
                if old_id not in remove_ids:
                    new_vocab_fb[token] = id_mapping[old_id]
            model["vocab"] = new_vocab_fb

    data["added_tokens"] = _rebuild_added_tokens(
        data.get("added_tokens", []), remove_ids, id_mapping
    )
    data["model"] = model
    return data


def update_tokenizer_config_json(
    config: dict,
    remove_ids: Set[int],
    id_mapping: Dict[int, int],
    new_vocab_size: int,
) -> dict:
    """Update a parsed ``tokenizer_config.json`` dict with renumbered special-token IDs.

    Exits with an error if any special token is in the removal list.
    """
    out = json.loads(json.dumps(config))

    # Scalar special-token ID fields
    for key in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "mask_token_id",
        "sep_token_id",
        "cls_token_id",
    ):
        val = out.get(key)
        if val is not None and val in remove_ids:
            print(f"Error: special token '{key}' (ID {val}) is marked for removal.")
            sys.exit(1)
        if val is not None and val in id_mapping:
            out[key] = id_mapping[val]

    # Dict-form special tokens, e.g. {"content": "<s>", "id": 1, ...}
    for key in (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "mask_token",
        "sep_token",
        "cls_token",
    ):
        entry = out.get(key)
        if isinstance(entry, dict) and "id" in entry:
            old_id = entry["id"]
            if old_id in remove_ids:
                print(f"Error: special token '{key}' with ID {old_id} is marked for removal.")
                sys.exit(1)
            if old_id in id_mapping:
                entry["id"] = id_mapping[old_id]

    out["vocab_size"] = new_vocab_size
    return out


def update_special_tokens_map_json(
    mapping: dict,
    remove_ids: Set[int],
    id_mapping: Dict[int, int],
) -> dict:
    """Update a parsed ``special_tokens_map.json`` dict with renumbered IDs."""
    out = json.loads(json.dumps(mapping))
    for key, entry in out.items():
        if isinstance(entry, dict) and "id" in entry:
            old_id = entry["id"]
            if old_id in remove_ids:
                print(
                    f"Error: special token '{key}' (ID {old_id}) "
                    f"is marked for removal."
                )
                sys.exit(1)
            if old_id in id_mapping:
                entry["id"] = id_mapping[old_id]
    return out


def _copy_extra_tokenizer_files(src_dir: str, dst_dir: str) -> None:
    """Copy auxiliary tokenizer files (e.g. ``tokenizer.model``) to *dst_dir*.

    Files already present in *dst_dir* (like tokenizer.json and
    tokenizer_config.json) are skipped.
    """
    skip = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    for name in os.listdir(src_dir):
        if name.startswith("tokenizer") and name not in skip:
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, name))


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def verify_pruned_vocab_size(
    input_dir: str,
    vocab_size: int,
    new_vocab_size: int,
) -> None:
    """Verify that every ``.safetensors`` shard has correctly pruned tensors.

    Checks that every vocab-mapped tensor now has *new_vocab_size* as its
    first dimension.  This catches any shard that was missed or incorrectly
    pruned.
    """
    safetensors_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".safetensors")
    )
    for fname in safetensors_files:
        tensors = load_file(os.path.join(input_dir, fname), device="cpu")
        for name, tensor in tensors.items():
            if is_vocab_tensor(name, tensor.size(), vocab_size):
                print(
                    f"Consistency error: tensor '{name}' in {fname} still "
                    f"has vocab_size={tensor.size(0)} (expected {new_vocab_size})."
                )
                sys.exit(1)


def check_tokenizer_consistency(
    tokenizer_data: dict,
    tokenizer_config: dict,
    new_vocab_size: int,
) -> None:
    """Verify tokenizer data matches the new vocabulary size and special-token IDs are valid."""
    # 1. Tokenizer vocabulary size matches
    model_type = tokenizer_data.get("model", {}).get("type", "")
    if model_type in ("BPE", "WordPiece"):
        tk_vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
    elif model_type in ("Unigram", "SentencePiece"):
        tk_vocab_size = len(tokenizer_data.get("model", {}).get("vocab", []))
    else:
        tk_vocab_size = -1

    if tk_vocab_size > new_vocab_size:
        print(
            f"Consistency error: tokenizer vocabulary size ({tk_vocab_size}) "
            f"exceeds model vocabulary size ({new_vocab_size})."
        )
        sys.exit(1)

    # 2. Special tokens exist with valid IDs
    special_ids: List[Optional[int]] = []
    for key in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        val = tokenizer_config.get(key)
        if val is not None:
            special_ids.append(val)

    for key in (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
    ):
        entry = tokenizer_config.get(key)
        if isinstance(entry, dict) and "id" in entry:
            special_ids.append(entry["id"])

    for sid in special_ids:
        if sid is not None and not (0 <= sid < new_vocab_size):
            print(
                f"Consistency error: special token ID {sid} "
                f"is out of range (0\u2013{new_vocab_size - 1})."
            )
            sys.exit(1)


_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
}


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes *tensor* occupies (numel × element size)."""
    return tensor.numel() * _DTYPE_BYTES.get(tensor.dtype, 4)


def audit_vocab_tensor_changes(
    input_dir: str,
    output_dir: str,
    vocab_size: int,
    new_vocab_size: int,
) -> None:
    """Audit every tensor whose first dimension was originally *vocab_size*.

    Scans all ``.safetensors`` shards in *input_dir* for tensors with
    ``shape[0] == vocab_size``, then locates the corresponding tensor in
    *output_dir*.  Prints a detailed table and overall savings summary.

    Exits with an error if any such tensor was **not** resized to
    *new_vocab_size*.
    """
    # ── Collect before/after info ──────────────────────────────────────
    records: List[dict] = []

    for fname in sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors")):
        src_tensors = load_file(os.path.join(input_dir, fname), device="cpu")
        for name, src_tensor in src_tensors.items():
            if src_tensor.ndim < 1 or src_tensor.size(0) != vocab_size:
                continue
            dst_tensor = None
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path):
                dst_tensors = load_file(out_path, device="cpu")
                dst_tensor = dst_tensors.get(name)
            records.append({
                "name": name,
                "shard": fname,
                "dtype": src_tensor.dtype,
                "old_shape": list(src_tensor.shape),
                "old_bytes": _tensor_bytes(src_tensor),
                "new_shape": list(dst_tensor.shape) if dst_tensor is not None else None,
                "new_bytes": _tensor_bytes(dst_tensor) if dst_tensor is not None else 0,
            })

    if not records:
        print("  (no tensors with first dim == vocab_size found)")
        return

    # ── Print table ────────────────────────────────────────────────────
    print(f"\n  {'Tensor':<56} {'Orig shape':<20} {'New shape':<20} {'Dtype':<14} {'Saved':>10}")
    print(f"  {'-'*56} {'-'*20} {'-'*20} {'-'*14} {'-'*10}")

    total_old = 0
    total_new = 0
    any_error = False

    for r in records:
        saved = r["old_bytes"] - r["new_bytes"]
        total_old += r["old_bytes"]
        total_new += r["new_bytes"]

        new_shape_str = str(r["new_shape"]) if r["new_shape"] is not None else "MISSING"
        name_short = f"{r['shard']}:{r['name']}" if len(r['name']) > 50 else r['name']
        print(f"  {name_short:<56} {str(r['old_shape']):<20} {new_shape_str:<20} {str(r['dtype']):<14} {_fmt_bytes(saved):>10}")

        if r["new_shape"] is None or r["new_shape"][0] != new_vocab_size:
            any_error = True
            print(
                f"  \u2716 Error: {r['name']} has {r['new_shape'][0] if r['new_shape'] else '?'} "
                f"rows, expected {new_vocab_size}",
            )

    # ── Totals ─────────────────────────────────────────────────────────
    print(f"  {'-'*56} {'-'*20} {'-'*20} {'-'*14} {'-'*10}")
    print(f"  {'TOTAL':<56} {_fmt_bytes(total_old):>20} {_fmt_bytes(total_new):>20} {'':<14} {_fmt_bytes(total_old - total_new):>10}")
    savings_pct = 100.0 * (total_old - total_new) / total_old if total_old else 0
    print(f"  Total checkpoint size before:  {_fmt_bytes(total_old, pad=10)}")
    print(f"  Total checkpoint size after:   {_fmt_bytes(total_new, pad=10)}")
    print(f"  Total bytes removed:           {_fmt_bytes(total_old - total_new, pad=10)}")
    print(f"  Reduction:                     {savings_pct:.2f}%")

    if any_error:
        sys.exit(1)


def _fmt_bytes(n: int, pad: int = 0) -> str:
    """Format *n* bytes as a human-readable string (e.g. ``1.23 GiB``)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:>{pad}.{2 if unit != 'B' else 0}f} {unit}" if pad else f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.2f} PiB"


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _copy_non_safetensors(input_dir: str, output_dir: str) -> None:
    """Copy files needed for the model to load.

    Skips tokenizer files (handled separately), safetensors files
    (already handled), config files (written explicitly), and the
    safetensors index (copied during pruning).
    """
    skip_exact = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    }
    for name in os.listdir(input_dir):
        if name in skip_exact:
            continue
        if name.endswith(".safetensors"):
            continue
        if name.startswith("tokenizer"):
            continue
        src = os.path.join(input_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))


def save_everything(
    output_dir: str,
    tokenizer_data: dict,
    tokenizer_config: dict,
    special_tokens_map: Optional[dict],
    input_dir: str,
) -> None:
    """Write pruned tokenizer files to *output_dir*.

    Safetensors weight files and config.json are already on disk
    (pruned/saved in earlier steps).  This function writes the
    tokenizer text-format files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # tokenizer.json
    print("  Saving tokenizer.json…")
    tk_json_path = os.path.join(output_dir, "tokenizer.json")
    with open(tk_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # tokenizer_config.json
    print("  Saving tokenizer_config.json…")
    tk_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tk_cfg_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    # special_tokens_map.json (if present in original)
    if special_tokens_map is not None:
        print("  Saving special_tokens_map.json…")
        stm_path = os.path.join(output_dir, "special_tokens_map.json")
        with open(stm_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    # Remaining tokenizer files (tokenizer.model, …)
    _copy_extra_tokenizer_files(input_dir, output_dir)

    # Model support files (config, generation_config, tokeniser files,
    # modelling code, etc.)
    _copy_non_safetensors(input_dir, output_dir)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_vocab_size(config) -> int:
    """
    Extract vocabulary size from various config types.

    Handles standard configs (vocab_size), MoE configs (text_config.vocab_size),
    and encoder-decoder configs (decoder.vocab_size or encoder.vocab_size).
    """
    # Direct attribute
    if hasattr(config, "vocab_size") and config.vocab_size is not None:
        return config.vocab_size

    # Nested configs (MoE, encoder-decoder, etc.)
    for attr in ("text_config", "decoder", "encoder", "config"):
        nested = getattr(config, attr, None)
        if nested is not None and hasattr(nested, "vocab_size"):
            return nested.vocab_size

    # Fallback: check if it's a dict-like config
    if hasattr(config, "to_dict"):
        d = config.to_dict()
        for key in ("vocab_size", "text_config.vocab_size", "decoder.vocab_size", "encoder.vocab_size"):
            val = d
            for part in key.split("."):
                val = val.get(part) if isinstance(val, dict) else None
                if val is None:
                    break
            if val is not None:
                return val

    raise AttributeError(
        f"Could not determine vocab_size from config of type {type(config).__name__}. "
        "Please check the config structure manually."
    )


def get_hidden_size(config) -> int:
    """Extract hidden size from various config types."""
    for attr in ("hidden_size", "n_embd", "d_model", "d_embed"):
        if hasattr(config, attr) and getattr(config, attr) is not None:
            return getattr(config, attr)

    for attr in ("text_config", "decoder", "encoder"):
        nested = getattr(config, attr, None)
        if nested is not None and hasattr(nested, "hidden_size"):
            return nested.hidden_size

    return 0


# ---------------------------------------------------------------------------
# Tensor explainer
# ---------------------------------------------------------------------------


class _TensorExplainer:
    """Return annotation dicts explaining the role of each tensor pattern."""

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB"):
            if abs(n) < 1024:
                return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
            n /= 1024
        return f"{n:.2f} TiB"

    @staticmethod
    def _norm_name(name: str) -> str:
        import re
        n = re.sub(r"layers\.\d+", "layers.N", name)
        n = re.sub(r"experts\.\d+", "experts.E", n)
        n = re.sub(r"blocks\.\d+", "blocks.N", n)
        return n

    @staticmethod
    def _has_quant_suffix(name: str) -> bool:
        return any(
            name.endswith(s) for s in (
                "_scale", "_scale_2", "input_scale",
            )
        )

    @staticmethod
    def classify(name: str, shape) -> dict:
        lower = name.lower()

        # ── Quantisation metadata ────────────────────────────────────────
        if _TensorExplainer._has_quant_suffix(name):
            sz = list(shape)
            is_per_group = len(sz) >= 1 and sz[-1] <= 128 and sz[-1] > 1
            return {
                "category": "Quantisation metadata",
                "purpose": (
                    "NVFP4 scale factor per-group (weight_scale) or "
                    "per-tensor (input_scale, weight_scale_2). "
                    "Dequantises the packed 4-bit weights."
                ),
                "formula": "w_deq = w_packed * scale",
                "used": "During dequantisation, before every matmul.",
                "vocab_dep": "No. Scales are per-channel or per-tensor.",
                "prunable": "Only if the associated weight is pruned.",
                "affects": "Inference efficiency (quantisation params).",
            }

        # ── Embedding ────────────────────────────────────────────────────
        if "embed_tokens" in lower or lower.endswith("wte.weight"):
            return {
                "category": "Embedding",
                "purpose": (
                    "Maps discrete token IDs to dense hidden vectors. "
                    "Each row is the learned representation of one token."
                ),
                "formula": "x = Embedding[token_id]",
                "used": "At the very start of every forward pass.",
                "vocab_dep": "Yes. First dimension == vocab_size.",
                "prunable": "Yes, by vocabulary pruning (removing token rows).",
                "affects": "Model capacity (embedding dimension per token).",
            }

        # ── LM head ──────────────────────────────────────────────────────
        if "lm_head" in lower and not _TensorExplainer._has_quant_suffix(name):
            qs = ""
            if shape and len(shape) >= 2 and shape[1] == 1024:
                qs = " (NVFP4 packed: 2 values per byte, original dim 2048)"
            return {
                "category": "LM Head",
                "purpose": (
                    "Projects the final hidden state to vocabulary logits. "
                    "Each row corresponds to one token's score."
                ),
                "formula": "logits = h * W_lm^T",
                "used": "At the final layer to compute next-token logits.",
                "vocab_dep": f"Yes. Shape[0] == vocab_size{qs}.",
                "prunable": "Yes, by vocabulary pruning (removing token rows).",
                "affects": "Model capacity (vocab projection size).",
            }

        # ── Self-attention ───────────────────────────────────────────────
        if "self_attn" in lower:
            sub = "Q" if "q_proj" in lower else "K" if "k_proj" in lower else "V" if "v_proj" in lower else "O" if "o_proj" in lower else "attention"
            desc = {
                "Q": ("Query", "Q = X W_q", "dot-product with K"),
                "K": ("Key", "K = X W_k", "dot-product with Q"),
                "V": ("Value", "V = X W_v", "weighted sum via attention scores"),
                "O": ("Output", "O = Attn W_o", "project concatenated heads back"),
            }.get(sub, ("attention", "standard attention", "attention mechanism"))
            return {
                "category": "Self-attention",
                "purpose": f"Projects hidden states into {desc[0]} vectors.",
                "formula": desc[1],
                "used": f"Every forward pass. Used in {desc[2]}.",
                "vocab_dep": "No. Shape is hidden_dim × head_dim.",
                "prunable": "By attention-head pruning or layer removal.",
                "affects": "Model capacity (more heads = more expressivity).",
            }

        if "self_attn" in lower and ("norm" in lower or "rms" in lower):
            return {
                "category": "Self-attention",
                "purpose": "Normalises Q or K vectors before attention (RoPE prep).",
                "formula": "Q = rms_norm(Q); K = rms_norm(K)",
                "used": "Every forward pass, before RoPE.",
                "vocab_dep": "No. Per-head scale.",
                "prunable": "Only with associated attention head.",
                "affects": "Inference stability, not capacity.",
            }

        # ── Linear attention ─────────────────────────────────────────────
        if "linear_attn" in lower:
            return {
                "category": "Linear attention",
                "purpose": (
                    "Implements a linear attention variant "
                    "(Mamba-2 / state-space dual form) projection."
                ),
                "formula": "SSM-based: y = SSM(x)",
                "used": "Every forward pass on alternating layers.",
                "vocab_dep": "No.",
                "prunable": "By layer removal only.",
                "affects": "Model capacity.",
            }

        # ── MoE Router ──────────────────────────────────────────────────
        if "router" in lower or "gate.weight" in lower:
            return {
                "category": "MoE Router",
                "purpose": (
                    "Computes routing logits that determine which experts "
                    "each token is dispatched to (top-k routing)."
                ),
                "formula": "routing_weights = softmax(x W_router)",
                "used": "Every forward pass, before expert computation.",
                "vocab_dep": "No. Shape is hidden_dim × num_experts.",
                "prunable": "Can shrink with expert count reduction.",
                "affects": "Routing policy; small fraction of total params.",
            }

        # ── MoE Expert (gate, up or down projection) ────────────────────
        if "experts." in lower or "expert." in lower:
            is_shared = "shared" in lower

            # Fused gate+up projection (DeepSeek-style MoE)
            if "gate_up_proj" in lower:
                return {
                    "category": "Shared Expert" if is_shared else "MoE Expert (fused Gate+Up)",
                    "purpose": (
                        "Fused gate and up projection in the SwiGLU FFN "
                        "inside each expert. Combines both projections "
                        "into one tensor for efficiency."
                    ),
                    "formula": "gate = silu(x W_gate); up = x W_up",
                    "used": "Every forward pass (activated per routed token).",
                    "vocab_dep": "No.",
                    "prunable": "By expert pruning (removing entire expert).",
                    "affects": "Model capacity.",
                }

            if "gate_proj" in lower:
                return {
                    "category": "Shared Expert" if is_shared else "MoE Expert Gate",
                    "purpose": (
                        "Gate projection in the SwiGLU FFN inside each "
                        f"{'shared' if is_shared else 'routed'} expert."
                    ),
                    "formula": "gate = silu(x W_gate)",
                    "used": "Every forward pass (activated per routed token).",
                    "vocab_dep": "No.",
                    "prunable": "By expert pruning (removing entire expert).",
                    "affects": "Model capacity.",
                }
            if "up_proj" in lower:
                return {
                    "category": "Shared Expert" if is_shared else "MoE Expert Up",
                    "purpose": (
                        "Up projection in the SwiGLU FFN inside each "
                        f"{'shared' if is_shared else 'routed'} expert."
                    ),
                    "formula": "up = x W_up",
                    "used": "Every forward pass (activated per routed token).",
                    "vocab_dep": "No.",
                    "prunable": "By expert pruning (removing entire expert).",
                    "affects": "Model capacity.",
                }
            if "down_proj" in lower:
                return {
                    "category": "Shared Expert" if is_shared else "MoE Expert Down",
                    "purpose": (
                        "Down projection in the SwiGLU FFN inside each "
                        f"{'shared' if is_shared else 'routed'} expert. "
                        "Combines gate*up back to hidden_dim."
                    ),
                    "formula": "down = (gate * up) W_down",
                    "used": "Every forward pass (activated per routed token).",
                    "vocab_dep": "No.",
                    "prunable": "By expert pruning (removing entire expert).",
                    "affects": "Model capacity.",
                }
            # Fallback for expert tensors that don't match known patterns
            return {
                "category": "MoE Expert",
                "purpose": "Part of a MoE expert's computation.",
                "formula": "varies",
                "used": "Every forward pass when expert is activated.",
                "vocab_dep": "No.",
                "prunable": "By expert pruning.",
                "affects": "Model capacity.",
            }

        # ── MLP (dense, non-expert) ──────────────────────────────────────
        if "mlp" in lower or "fc" in lower:
            return {
                "category": "MLP",
                "purpose": "Feed-forward network projection (up/down or FC1/FC2).",
                "formula": "MLP(x) = down(silu(up(x)))",
                "used": "Every forward pass after attention.",
                "vocab_dep": "No.",
                "prunable": "By layer removal or width reduction.",
                "affects": "Model capacity.",
            }

        # ── Layer norms ──────────────────────────────────────────────────
        if "norm" in lower and ("weight" in name or "bias" in name):
            return {
                "category": "Layer Norm",
                "purpose": (
                    "Element-wise scale (and bias) for layer normalisation. "
                    "Stabilises training and inference."
                ),
                "formula": "y = gamma * (x - mean) / std + beta",
                "used": "Every forward pass, before/after each sub-layer.",
                "vocab_dep": "No. Shape is [hidden_dim].",
                "prunable": "Only with associated hidden dim change.",
                "affects": "Inference stability.",
            }

        # ── Vision encoder ──────────────────────────────────────────────
        if "visual" in lower or "vision" in lower:
            cat = "Vision encoder"
            if "patch_embed" in lower:
                purp = "Converts image patches into visual tokens."
            elif "pos_embed" in lower:
                purp = "Learned positional embedding for visual tokens."
            elif "merger" in lower:
                purp = "Projects visual tokens into language-model hidden space."
            elif "attn" in lower or "attention" in lower:
                purp = "Self-attention in the vision encoder."
            elif "mlp" in lower or "fc" in lower:
                purp = "Feed-forward network in the vision encoder."
            elif "norm" in lower:
                purp = "Layer norm in the vision encoder."
            else:
                purp = "Vision encoder component."
            return {
                "category": cat,
                "purpose": purp,
                "formula": "varies",
                "used": "Only for vision inputs (multi-modal inference).",
                "vocab_dep": "No.",
                "prunable": "Only by removing the vision encoder entirely.",
                "affects": "Model capacity (vision modality).",
            }

        # ── MTP (Multi-Token Prediction) ────────────────────────────────
        if lower.startswith("mtp"):
            return {
                "category": "MTP",
                "purpose": (
                    "Multi-Token Prediction module. "
                    "Predicts K future tokens in parallel at each position."
                ),
                "formula": "varies (transformer decoder + expert MoE)",
                "used": "During training and speculative decoding.",
                "vocab_dep": "No.",
                "prunable": "Only by removing MTP module entirely.",
                "affects": "Training efficiency / speculative decoding speed.",
            }

        # ── Fallback ─────────────────────────────────────────────────────
        return {
            "category": "Other",
            "purpose": "Unknown or unclassified tensor.",
            "formula": "—",
            "used": "varies",
            "vocab_dep": "Cannot determine.",
            "prunable": "Unknown.",
            "affects": "Unknown.",
        }


def explain_tensors(input_dir: str) -> None:
    """Print a detailed explanation for every unique tensor pattern."""
    import re
    import safetensors

    # ── Collect patterns ──────────────────────────────────────────────
    groups: dict = {}
    total_all = 0

    safetensors_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        print("Error: no .safetensors files found.")
        sys.exit(1)

    for fname in safetensors_files:
        with safetensors.safe_open(
            os.path.join(input_dir, fname), framework="pt"
        ) as f:
            for name in f.keys():
                shape = f.get_tensor(name).shape
                t = f.get_tensor(name)
                nbytes = t.numel() * t.element_size()
                total_all += nbytes

                norm = _TensorExplainer._norm_name(name)
                # Count unique layers/experts in this group
                m_layers = set(re.findall(r"layers\.(\d+)", name))
                m_experts = set(re.findall(r"experts\.(\d+)", name))
                m_blocks = set(re.findall(r"blocks\.(\d+)", name))

                if norm not in groups:
                    groups[norm] = {
                        "example": name,
                        "shape": list(shape),
                        "dtype": str(t.dtype).split(".")[-1],
                        "count": 0,
                        "total_bytes": 0,
                        "layers": set(),
                        "experts": set(),
                        "blocks": set(),
                        "nbytes_single": nbytes,
                    }
                g = groups[norm]
                g["count"] += 1
                g["total_bytes"] += nbytes
                g["layers"] |= m_layers
                g["experts"] |= m_experts
                g["blocks"] |= m_blocks

    # ── Sort by total bytes descending ────────────────────────────────
    sorted_groups = sorted(groups.items(), key=lambda kv: -kv[1]["total_bytes"])

    # ── Print ─────────────────────────────────────────────────────────
    for norm_name, g in sorted_groups:
        ann = _TensorExplainer.classify(norm_name, g["shape"])

        # Build a compact instance range string
        ranges = []
        if g["layers"]:
            nums = sorted(int(x) for x in g["layers"])
            ranges.append(f"layers {nums[0]}..{nums[-1]}")
        if g["experts"]:
            nums = sorted(int(x) for x in g["experts"])
            ranges.append(f"experts {nums[0]}..{nums[-1]}")
        if g["blocks"]:
            nums = sorted(int(x) for x in g["blocks"])
            ranges.append(f"blocks {nums[0]}..{nums[-1]}")

        range_str = f"  ({', '.join(ranges)})" if ranges else ""
        count_str = f"{g['count']} occurrences"
        size_str = _TensorExplainer._fmt_bytes(g["total_bytes"])

        print(f"\nTensor pattern:")
        print(f"  {norm_name}  [{', '.join(str(d) for d in g['shape'])}]  {g['dtype']}")
        print(f"  {count_str}, {size_str} total{range_str}")
        w = g["nbytes_single"]
        if g["count"] > 1:
            print(f"  ({_TensorExplainer._fmt_bytes(w)} per instance)")

        print(f"\n  Category:  {ann['category']}")
        print(f"  Purpose:   {ann['purpose']}")
        print(f"  Formula:   {ann['formula']}")
        print(f"  Used:      {ann['used']}")
        print(f"  Prunable:  {ann['prunable']}")
        print(f"  Vocab dep: {ann['vocab_dep']}")
        print(f"  Affects:   {ann['affects']}")
        print(f"\n  {'─' * 60}")

    print(f"\nTotal checkpoint size: {_TensorExplainer._fmt_bytes(total_all)}")
    print(f"Unique tensor patterns: {len(groups)}")
    print(f"Total tensors: {sum(g['count'] for _, g in sorted_groups)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    remove_ids = parse_remove_ids(args.remove_ids)
    if args.remove_ids_file:
        try:
            with open(args.remove_ids_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        remove_ids.add(int(line))
                    except ValueError:
                        print(
                            f"Error: '{line}' in --remove-ids-file is not a "
                            "valid integer token ID.",
                        )
                        sys.exit(1)
        except OSError as exc:
            print(f"Error: cannot read --remove-ids-file: {exc}")
            sys.exit(1)
    input_dir = args.model
    output_dir = args.output
    dry_run = args.dry_run

    # ── Explain-tensors mode ─────────────────────────────────────────────
    if args.explain_tensors:
        explain_tensors(input_dir)
        return

    # ── Validate required arguments ───────────────────────────────────────
    if not args.output:
        print("Error: --output is required (unless using --explain-tensors).")
        sys.exit(1)
    output_dir = args.output

    # Deferred import: safetensors is only needed for saving, not for --help
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("Error: safetensors is required. Install it with: pip install safetensors")
        sys.exit(1)

    if not remove_ids:
        print(
            "Error: no token IDs to remove. "
            "Use --remove-ids or --remove-ids-file.",
        )
        sys.exit(1)

    # ── 1. Load config & tokenizer (no model loading) ─────────────────────
    print(f"Loading config from '{input_dir}'…")
    config = AutoConfig.from_pretrained(input_dir)
    print(f"Loading tokenizer from '{input_dir}'…")
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    original_vocab_size = get_vocab_size(config)

    # ── 2. Validate token IDs ──────────────────────────────────────────────
    print(f"\nOriginal vocabulary size: {original_vocab_size}")
    if len(remove_ids) <= 20:
        print(f"Token IDs to remove ({len(remove_ids)}): {sorted(remove_ids)}")
    else:
        print(f"Token IDs to remove ({len(remove_ids)}): {sorted(remove_ids)[:10]} ... {sorted(remove_ids)[-10:]}")
    validate_token_ids(remove_ids, original_vocab_size)

    # ── 3. Build mapping ───────────────────────────────────────────────────
    kept_ids = build_kept_ids(remove_ids, original_vocab_size)
    id_mapping = build_id_mapping(kept_ids)
    new_vocab_size = len(kept_ids)
    num_removed = original_vocab_size - new_vocab_size
    reduction_pct = 100.0 * num_removed / original_vocab_size

    print(f"New vocabulary size:      {new_vocab_size}")
    print(f"Tokens removed:           {num_removed}")
    print(f"Reduction:                {reduction_pct:.2f}%")
    print(f"Tokenizer type:           {detect_tokenizer_type(tokenizer)}")

    # Detect tied embeddings from safetensors (lm_head.weight absent means tied)
    safetensors_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".safetensors")
    )
    has_lm_head = False
    has_embed = False
    for fname in safetensors_files:
        tensors = load_file(os.path.join(input_dir, fname), device="cpu")
        has_lm_head = has_lm_head or any(
            "lm_head" in k and k.endswith(".weight") for k in tensors
        )
        has_embed = has_embed or any(
            is_vocab_tensor(k, tensors[k].size(), original_vocab_size)
            and "lm_head" not in k
            for k in tensors
        )
    tied = not has_lm_head or (has_lm_head and not has_embed)
    print(f"Tied embeddings:          {tied}")

    if dry_run:
        print("\n✔ Dry-run validation passed.  No files were written.")
        return

    if os.path.exists(output_dir):
        print(f"Error: output directory '{output_dir}' already exists.")
        sys.exit(1)
    os.makedirs(output_dir)

    # ── 4. Prune safetensors weights ─────────────────────────────────────
    print("\nPruning vocabulary in safetensors files…")
    prune_safetensors_directory(
        input_dir, output_dir, remove_ids, original_vocab_size,
        id_mapping, new_vocab_size,
    )

    # Save config.json immediately (so it exists even if later steps fail)
    print("  Saving config.json…")
    config_src = os.path.join(input_dir, "config.json")
    with open(config_src, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["vocab_size"] = new_vocab_size
    config_dst = os.path.join(output_dir, "config.json")
    with open(config_dst, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # ── 5. Rebuild tokenizer files ────────────────────────────────────────
    print("Rebuilding tokenizer files…")

    tk_json_src = os.path.join(input_dir, "tokenizer.json")
    if not os.path.exists(tk_json_src):
        print("Error: tokenizer.json not found in model directory.")
        sys.exit(1)

    with open(tk_json_src, "r", encoding="utf-8") as f:
        raw_tk_data = json.load(f)

    new_tk_data = rebuild_tokenizer_json(raw_tk_data, remove_ids, id_mapping)

    tk_cfg_src = os.path.join(input_dir, "tokenizer_config.json")
    if os.path.exists(tk_cfg_src):
        with open(tk_cfg_src, "r", encoding="utf-8") as f:
            raw_tk_cfg = json.load(f)
        new_tk_cfg = update_tokenizer_config_json(
            raw_tk_cfg, remove_ids, id_mapping, new_vocab_size
        )
    else:
        new_tk_cfg = {"vocab_size": new_vocab_size}

    stm_data: Optional[dict] = None
    stm_src = os.path.join(input_dir, "special_tokens_map.json")
    if os.path.exists(stm_src):
        with open(stm_src, "r", encoding="utf-8") as f:
            stm_data = update_special_tokens_map_json(
                json.load(f), remove_ids, id_mapping
            )

    # ── 6. Consistency checks ─────────────────────────────────────────────
    print("Running consistency checks…")

    verify_pruned_vocab_size(output_dir, original_vocab_size, new_vocab_size)
    print("  All safetensors tensors correctly pruned \u2714")

    check_tokenizer_consistency(new_tk_data, new_tk_cfg, new_vocab_size)
    print("  Tokenizer data consistent \u2714")

    # ── 7. Final verification ──────────────────────────────────────────────
    print("Running final verification…")

    # 7a. Full audit of every vocab-indexed tensor
    audit_vocab_tensor_changes(
        input_dir, output_dir, original_vocab_size, new_vocab_size,
    )

    # 7b. Load rebuilt tokenizer and test roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tk_json = os.path.join(tmpdir, "tokenizer.json")
        with open(tmp_tk_json, "w", encoding="utf-8") as f:
            json.dump(new_tk_data, f, ensure_ascii=False)
        _copy_extra_tokenizer_files(input_dir, tmpdir)

        tmp_tk_cfg = os.path.join(tmpdir, "tokenizer_config.json")
        with open(tmp_tk_cfg, "w", encoding="utf-8") as f:
            json.dump(new_tk_cfg, f, ensure_ascii=False)

        if stm_data is not None:
            tmp_stm = os.path.join(tmpdir, "special_tokens_map.json")
            with open(tmp_stm, "w", encoding="utf-8") as f:
                json.dump(stm_data, f, ensure_ascii=False)

        try:
            tk = AutoTokenizer.from_pretrained(tmpdir, use_fast=True)
            print("  Rebuilt tokenizer loads successfully \u2714")
        except Exception as exc:
            print(
                f"Error: rebuilt tokenizer could not be loaded: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Tokenizer vocab_size must not exceed model vocab_size
        # (it may be smaller if some model vocab slots are unused)
        tk_vocab = getattr(tk, "vocab_size", None) or len(tk)
        if tk_vocab > new_vocab_size:
            print(
                f"Error: tokenizer vocab_size ({tk_vocab}) "
                f"> model vocab_size ({new_vocab_size})."
            )
            sys.exit(1)
        if tk_vocab < new_vocab_size:
            print(
                f"  tokenizer vocab_size ({tk_vocab}) < model "
                f"vocab_size ({new_vocab_size}) "
                f"({new_vocab_size - tk_vocab} unused slots) \u2714"
            )
        else:
            print(f"  tokenizer.vocab_size == {new_vocab_size} \u2714")

        # Tokenization roundtrip: IDs should be within range
        for test_text in ("hello", " world", "привет", "test sentence"):
            ids = tk.encode(test_text)
            bad = [i for i in ids if not (0 <= i < new_vocab_size)]
            if bad:
                print(
                    f"Error: tokenizer({test_text!r}) returned "
                    f"out-of-range IDs: {bad}"
                )
                sys.exit(1)
        print("  Tokenization roundtrip returns valid IDs \u2714")

    # ── 8. Save everything ─────────────────────────────────────────────────
    print(f"Saving pruned model to '{output_dir}'…")
    save_everything(
        output_dir,
        new_tk_data,
        new_tk_cfg,
        stm_data,
        input_dir,
    )

    # ── 9. Post-save sweep: no tensor should still use original vocab_size ─
    print("Post-save sweep: checking for stray tensors with original vocab dimension…")
    stray: List[str] = []
    for fname in sorted(f for f in os.listdir(output_dir) if f.endswith(".safetensors")):
        tensors = load_file(os.path.join(output_dir, fname), device="cpu")
        for name, tensor in tensors.items():
            if tensor.ndim >= 1 and tensor.size(0) == original_vocab_size:
                stray.append(f"  {fname}:{name} {list(tensor.shape)}")
    if stray:
        print("Error: the following tensors still have first dim == original_vocab_size:")
        for s in stray:
            print(s)
        sys.exit(1)
    print("  All clear — no tensor retains the original vocab dimension \u2714")


    # Spot-check skipped for quantized models (modelopt etc.)
    # Load with the target framework (VLLM / TensorRT-LLM) to verify.
    print("\nDone.  Summary:")
    print(f"  Original vocab size:  {original_vocab_size}")
    print(f"  New vocab size:        {new_vocab_size}")
    print(f"  Removed tokens:        {num_removed}")
    print(f"  Reduction:             {reduction_pct:.2f}%")


if __name__ == "__main__":
    main()
