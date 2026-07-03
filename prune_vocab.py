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
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Set

import safetensors.torch  # noqa: F401  (ensures safetensors is importable)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        required=True,
        help="Path where the pruned model will be saved.",
    )
    parser.add_argument(
        "--remove-ids",
        required=True,
        help="Comma-separated list of token IDs to remove (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report what would change without writing files.",
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
# Model helpers
# ---------------------------------------------------------------------------


def get_embed_tokens(model) -> torch.nn.Embedding:
    """Return the token embedding layer via the canonical HF accessor."""
    embed = model.get_input_embeddings()
    if embed is None:
        print("Error: model has no input embeddings.")
        sys.exit(1)
    return embed


def get_lm_head(model) -> torch.nn.Linear:
    """Return the LM output projection via the canonical HF accessor."""
    head = model.get_output_embeddings()
    if head is None:
        print("Error: model has no output embeddings (lm_head).")
        sys.exit(1)
    return head


def embeddings_are_tied(model) -> bool:
    """Return ``True`` if lm_head and embed_tokens share their weight storage."""
    embed = get_embed_tokens(model)
    head = get_lm_head(model)
    return head.weight.data_ptr() == embed.weight.data_ptr()


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


def perform_consistency_checks(
    model,
    new_vocab_size: int,
    id_mapping: Dict[int, int],
    tokenizer_data: dict,
    tokenizer_config: dict,
) -> None:
    """Run pre-save consistency checks and exit on failure."""
    embed = get_embed_tokens(model)
    head = get_lm_head(model)

    # 1. Embedding rows equal new vocabulary size
    if embed.weight.size(0) != new_vocab_size:
        print(
            f"Consistency error: embedding rows ({embed.weight.size(0)}) "
            f"!= new vocabulary size ({new_vocab_size})."
        )
        sys.exit(1)

    # 2. LM head rows equal new vocabulary size
    if head.weight.size(0) != new_vocab_size:
        print(
            f"Consistency error: lm_head rows ({head.weight.size(0)}) "
            f"!= new vocabulary size ({new_vocab_size})."
        )
        sys.exit(1)

    # 3. Tokenizer vocabulary size matches model vocabulary size
    model_type = tokenizer_data.get("model", {}).get("type", "")
    if model_type in ("BPE", "WordPiece"):
        tk_vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
    elif model_type in ("Unigram", "SentencePiece"):
        tk_vocab_size = len(tokenizer_data.get("model", {}).get("vocab", []))
    else:
        tk_vocab_size = -1

    if tk_vocab_size != new_vocab_size:
        print(
            f"Consistency error: tokenizer vocabulary size ({tk_vocab_size}) "
            f"!= model vocabulary size ({new_vocab_size})."
        )
        sys.exit(1)

    # 4. Special tokens exist with valid IDs
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

    # Also check dict-form special tokens
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


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_everything(
    model,
    config,
    output_dir: str,
    tokenizer_data: dict,
    tokenizer_config: dict,
    special_tokens_map: Optional[dict],
    input_dir: str,
) -> None:
    """Write pruned model, tokenizer, and configuration to *output_dir*.

    This function writes files directly rather than relying on
    ``tokenizer.save_pretrained()`` to have full control over the
    tokenizer file contents.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Model weights (safe_serialization=True → safetensors)
    print("  Saving model weights (safe_serialization)…")
    model.save_pretrained(output_dir, safe_serialization=True)

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

    # Ensure config.json is also saved (already done by model.save_pretrained,
    # but this guarantees the updated vocab_size is persisted even if the
    # model's save mechanism skips it for some reason).
    config.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    remove_ids = parse_remove_ids(args.remove_ids)
    input_dir = args.model
    output_dir = args.output
    dry_run = args.dry_run

    if not remove_ids:
        print("Error: --remove-ids is empty or contains no valid IDs.")
        sys.exit(1)

    # ── 1. Load model & tokenizer ──────────────────────────────────────────
    print(f"Loading model from '{input_dir}'…")
    model = AutoModelForCausalLM.from_pretrained(input_dir)
    print(f"Loading tokenizer from '{input_dir}'…")
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    config = model.config
    original_vocab_size = config.vocab_size

    # ── 2. Validate token IDs ──────────────────────────────────────────────
    print(f"\nOriginal vocabulary size: {original_vocab_size}")
    print(f"Token IDs to remove ({len(remove_ids)}): {sorted(remove_ids)}")
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
    print(f"Tied embeddings:          {embeddings_are_tied(model)}")

    if dry_run:
        print("\n✔ Dry-run validation passed.  No files were written.")
        return

    # ── 4. Prune model weights ─────────────────────────────────────────────
    print("\nPruning model weights…")
    embed = get_embed_tokens(model)
    head = get_lm_head(model)
    tied = embeddings_are_tied(model)

    # Embedding
    new_embed = prune_embedding_weight(embed, id_mapping, new_vocab_size)
    embed.weight.data = new_embed

    if not tied:
        new_head = prune_lm_head_weight(head, id_mapping, new_vocab_size)
        head.weight.data = new_head
    else:
        # When weights are tied, lm_head.weight IS embed_tokens.weight
        # (same Python object).  Updating embed_tokens already updated lm_head.
        print("  (lm_head is tied to embed_tokens \u2014 already updated)")

    # Update config
    config.vocab_size = new_vocab_size

    # ── 5. Rebuild tokenizer files ────────────────────────────────────────
    print("Rebuilding tokenizer files…")

    # tokenizer.json
    tk_json_src = os.path.join(input_dir, "tokenizer.json")
    if not os.path.exists(tk_json_src):
        print("Error: tokenizer.json not found in model directory.")
        sys.exit(1)

    with open(tk_json_src, "r", encoding="utf-8") as f:
        raw_tk_data = json.load(f)

    new_tk_data = rebuild_tokenizer_json(raw_tk_data, remove_ids, id_mapping)

    # tokenizer_config.json
    tk_cfg_src = os.path.join(input_dir, "tokenizer_config.json")
    if os.path.exists(tk_cfg_src):
        with open(tk_cfg_src, "r", encoding="utf-8") as f:
            raw_tk_cfg = json.load(f)
        new_tk_cfg = update_tokenizer_config_json(
            raw_tk_cfg, remove_ids, id_mapping, new_vocab_size
        )
    else:
        new_tk_cfg = {"vocab_size": new_vocab_size}

    # special_tokens_map.json
    stm_data: Optional[dict] = None
    stm_src = os.path.join(input_dir, "special_tokens_map.json")
    if os.path.exists(stm_src):
        with open(stm_src, "r", encoding="utf-8") as f:
            stm_data = update_special_tokens_map_json(
                json.load(f), remove_ids, id_mapping
            )

    # ── 6. Consistency checks ─────────────────────────────────────────────
    print("Running consistency checks…")
    perform_consistency_checks(
        model, new_vocab_size, id_mapping, new_tk_data, new_tk_cfg
    )

    # Also verify the rebuilt tokenizer can actually be loaded by
    # AutoTokenizer by writing to a temporary directory.
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
            _ = AutoTokenizer.from_pretrained(tmpdir, use_fast=True)
            print("  Rebuilt tokenizer loads successfully \u2714")
        except Exception as exc:
            print(
                f"Error: rebuilt tokenizer could not be loaded: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── 7. Save everything ─────────────────────────────────────────────────
    print(f"Saving pruned model to '{output_dir}'…")
    save_everything(
        model,
        config,
        output_dir,
        new_tk_data,
        new_tk_cfg,
        stm_data,
        input_dir,
    )

    # ── 8. Final spot-check ────────────────────────────────────────────────
    try:
        _ = AutoModelForCausalLM.from_pretrained(output_dir)
        _ = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
        print("  Saved model loads correctly \u2714")
    except Exception as exc:
        print(
            f"Warning: saved model could not be loaded back: {exc}",
            file=sys.stderr,
        )

    print("\nDone.  Summary:")
    print(f"  Original vocab size:  {original_vocab_size}")
    print(f"  New vocab size:        {new_vocab_size}")
    print(f"  Removed tokens:        {num_removed}")
    print(f"  Reduction:             {reduction_pct:.2f}%")


if __name__ == "__main__":
    main()
