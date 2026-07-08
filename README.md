# Prune Vocabulary

Permanently removes specified vocabulary token IDs from a Hugging Face causal language model (LLM) and writes out a pruned model that can be loaded directly with `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()` — no additional modifications needed.

## Why

Reducing vocabulary size lets you:

- **Shrink model size** — smaller embedding and LM-head matrices mean fewer parameters and a smaller disk footprint.
- **Cut memory usage** — lower `vocab_size` reduces peak VRAM/RAM during both training and inference.
- **Specialise a model** — strip unused tokens (e.g. non-English scripts, rare characters) before fine-tuning on a narrow domain.

## How it works

```
                          ┌──────────────────┐
                          │  Input model dir  │
                          │  config.json      │
                          │  tokenizer.json   │
                          │  tokenizer_config │
                          │  model.safetensors │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  AutoModelForCausalLM        │
                    │         +                    │
                    │  AutoTokenizer               │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  1. Validate token IDs       │
                    │  2. Build old→new ID mapping │
                    │  3. Prune embed_tokens       │
                    │  4. Prune lm_head            │
                    │  5. Rebuild tokenizer.json   │
                    │  6. Update configs           │
                    │  7. Consistency checks       │
                    └──────────────┬───────────────┘
                                   │
                          ┌────────┴─────────┐
                          │  Output model dir │
                          │  config.json      │
                          │  tokenizer.json   │
                          │  tokenizer_config │
                          │  model.safetensors │
                          └──────────────────┘
```

The script never retrains the tokenizer. Instead it rebuilds the `tokenizer.json` file by removing vocabulary entries and renumbering the remaining IDs consecutively from zero, preserving the original normalizer, pre-tokenizer, decoder, post-processor, and added tokens.

## Requirements

- Python ≥ 3.11
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers) (≥ 4.30)
- [safetensors](https://github.com/huggingface/safetensors)

These are the packages you would normally have in any HF model environment. CPU-only execution is fully supported.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic

```bash
python prune_vocab.py \
    --model ./llama-7b \
    --output ./llama-7b-pruned \
    --remove-ids 0,1,2
```

### Dry run (validate without writing)

```bash
python prune_vocab.py \
    --model ./llama-7b \
    --output ./llama-7b-pruned \
    --remove-ids 0,1,2 \
    --dry-run
```

### Remove many IDs

```bash
# Remove a contiguous range
python prune_vocab.py \
    --model ./model \
    --output ./model-pruned \
    --remove-ids "$(seq -s, 30000 32000)"

# Remove specific IDs
python prune_vocab.py \
    --model ./model \
    --output ./model-pruned \
    --remove-ids 10,20,30,40,50
```

### Text file (one ID per line)

```bash
python prune_vocab.py \
    --model ./model \
    --output ./model-pruned \
    --remove-ids-file ids.txt
```

### Combined sources (IDs merged)

```bash
python prune_vocab.py \
    --model ./model \
    --output ./model-pruned \
    --remove-ids "100001" \
    --remove-ids-file more_ids.txt
```

### Inspect every tensor in the checkpoint

```bash
python prune_vocab.py \
    --model ./model \
    --explain-tensors
```

Prints a detailed explanation for every unique tensor pattern in the model:
its shape, dtype, total size, category, mathematical purpose, formula,
when it is used during inference, whether it depends on vocabulary size,
whether it can be pruned, and whether it affects model capacity or only
inference efficiency.  No model or tokenizer loading is required.

### Load the pruned model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./llama-7b-pruned")
tokenizer = AutoTokenizer.from_pretrained("./llama-7b-pruned")

# Use as normal
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs)
```

## Arguments

| Argument                    | Required | Description                                                      |
|-----------------------------|----------|------------------------------------------------------------------|
| `--model`                   | Yes      | Path to the input Hugging Face model directory.                  |
| `--output`                  | Yes      | Path where the pruned model will be saved.                       |
| `--remove-ids`              | No*      | Comma-separated list of token IDs to remove.                     |
| `--remove-ids-file`         | No*      | Path to a text file with one token ID per line to remove.        |
| `--dry-run`                 | No       | Validate everything and report changes without writing.          |
| `--ignore-mismatched-sizes` | No       | Pass `ignore_mismatched_sizes=True` to `from_pretrained`. Use when loading quantized or custom models whose weight shapes differ from the architecture config. |
| `--explain-tensors`         | No       | Print detailed explanations for every tensor pattern in the checkpoint. Requires only `--model`; skips all pruning logic. |

\* At least one of `--remove-ids` or `--remove-ids-file` is required (unless using `--explain-tensors`).

## Step-by-step

1. **Load** — the model with `AutoModelForCausalLM.from_pretrained()` and its tokenizer with `AutoTokenizer.from_pretrained()`.

2. **Validate** — every ID from `--remove-ids` and/or `--remove-ids-file` is checked to be within `[0, vocab_size)`. The script exits immediately with a clear message if any ID is out of range, including special tokens (`bos_token_id`, `eos_token_id`, `pad_token_id`, `unk_token_id`, etc.).

3. **Map** — a set of kept IDs is built (all IDs not in the removal list), and an `old_id → new_id` mapping assigns each kept token a new consecutive ID starting at zero.

4. **Prune embeddings** — `model.get_input_embeddings().weight` (typically `model.model.embed_tokens`) is replaced with a new tensor containing only the kept rows, reordered according to the mapping. Uses `torch.index_select` (vectorised, memory-efficient). The same is done for `model.get_output_embeddings().weight` (`lm_head`) unless the weights are tied.

5. **Tied embeddings** — if `lm_head.weight` and `embed_tokens.weight` share the same storage (`data_ptr`), only the embedding tensor is pruned; the LM head automatically follows because it points to the same tensor object.

6. **Rebuild tokenizer** — the `tokenizer.json` file is loaded and its model vocabulary is modified:
   - **BPE / WordPiece** — `model.vocab` is a `{token: id}` dict. Entries whose ID is in the removal list are dropped, and the remaining keys are assigned new IDs.
   - **Unigram / SentencePiece** — `model.vocab` is a `[[token, score], ...]` list where the index is the ID. Entries at removed indices are dropped from the list.
   - The `added_tokens` list is similarly filtered and renumbered.
   - All other components (normalizer, pre-tokenizer, decoder, post-processor, truncation, padding) are preserved verbatim.

7. **Update configs** — `config.vocab_size` is updated; `tokenizer_config.json` and `special_tokens_map.json` have their special-token IDs renumbered. Any special token in the removal list triggers an error.

8. **Consistency checks** — before saving:
   - Embedding row count equals new vocabulary size
   - LM head row count equals new vocabulary size
   - Tokenizer vocabulary size equals model vocabulary size
   - All special token IDs fall within `[0, new_vocab_size)`
   - The rebuilt tokenizer can be loaded by `AutoTokenizer` (verified via a temp directory)

9. **Save** — the pruned model is saved with `safe_serialization=True` (safetensors), along with the rebuilt `tokenizer.json`, updated `tokenizer_config.json`, `special_tokens_map.json`, and any auxiliary tokenizer files (`tokenizer.model`, etc.).

10. **Spot-check** — after saving, the script immediately loads the output directory to confirm it works.

## Tokenizer type support

| Type              | vocab format in tokenizer.json                     | Models                                  |
|-------------------|----------------------------------------------------|-----------------------------------------|
| **BPE**           | `{token: id}` dict                                 | GPT-2, CodeGen, LLaMA (modern), Qwen    |
| **WordPiece**     | `{token: id}` dict                                 | BERT, DistilBERT, ELECTRA               |
| **Unigram**       | `[[token, score], ...]` list                       | ALBERT, XLNet, T5                       |
| **SentencePiece** | `[[token, score], ...]` list                       | LLaMA (via Unigram), Mistral, Gemma     |

Detection is automatic — the script reads the `model.type` field in `tokenizer.json`.

### Fallback for unknown types

If the model type is not one of the four above, the script prints a warning and attempts dict-style pruning. Unknown tokenizer types may still work but have not been tested.

## Output structure

```
output_dir/
├── config.json                 # updated vocab_size
├── tokenizer.json              # vocab pruned and renumbered
├── tokenizer_config.json       # special token IDs updated
├── special_tokens_map.json     # special tokens updated (if present)
├── model.safetensors           # pruned model weights
├── model.safetensors.index.json  # (if sharded)
└── tokenizer.model             # copied from input (SentencePiece binary, unchanged)
```

## Limitations

- **Requires `tokenizer.json`** — the script needs the Hugging Face tokenizers JSON format. Models that only provide a legacy `tokenizer.model` (SentencePiece binary) without `tokenizer.json` are not supported.
- **SentencePiece binary not updated** — the `tokenizer.model` binary file is copied to the output directory as-is. Loading with a *non-fast* tokenizer (e.g. `LlamaTokenizer` instead of `LlamaTokenizerFast`) may use the old vocabulary. Modern `AutoTokenizer` defaults to the fast version and reads from `tokenizer.json`, so this is not normally an issue.
- **Causal LM only** — the script is designed for `AutoModelForCausalLM`. Encoder-only or encoder-decoder models require different handling for their embedding layers.
- **No retraining** — the tokenization algorithm is preserved but the model's weights are not adapted to the new vocabulary. You should fine-tune the pruned model before using it for your task.
- **Model config may need tuning** — some model configurations reference `vocab_size` in other places (e.g. `gpt2`'s hard-coded `n_ctx` calculations). The script updates `config.vocab_size` which is sufficient for standard loading, but unusual architectures may need additional manual config changes.

## Error messages

| Scenario                                  | Message                                                        |
|-------------------------------------------|----------------------------------------------------------------|
| Invalid integer in `--remove-ids` / `--remove-ids-file` | `Error: 'abc' is not a valid integer token ID.`                |
| ID out of range                           | `Error: token ID 99999 is out of range. Vocabulary size is ...`|
| Special token in removal list             | `Error: special token 'bos_token_id' (ID 1) is marked for removal.` |
| `safetensors` not installed               | `Error: safetensors is required. Install it with: pip install safetensors` |
| `from_pretrained` fails (quantized model) | `RuntimeError: You set ignore_mismatched_sizes to False…` — retry with `--ignore-mismatched-sizes` |
| `tokenizer.json` missing                  | `Error: tokenizer.json not found in model directory.`          |
| Consistency check fails (embedding rows)  | `Consistency error: embedding rows ... != new vocabulary size ...` |
| Consistency check fails (tokenizer load)  | `Error: rebuilt tokenizer could not be loaded: ...`            |

## Testing

```bash
# Syntax check
python3 -c "import ast; ast.parse(open('prune_vocab.py').read()); print('OK')"

# Unit-test helper functions (no model load needed)
python3 -c "
from prune_vocab import build_kept_ids, build_id_mapping, prune_weight
import torch

kept = build_kept_ids({1,3}, 5)       # [0, 2, 4]
mapping = build_id_mapping([0, 2, 4])  # {0: 0, 2: 1, 4: 2}

w = torch.randn(5, 10)
pw = prune_weight(w, mapping, 3)
assert pw.shape == (3, 10)
print('OK')
"
```

## License

MIT
