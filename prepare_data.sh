#!/usr/bin/env bash
# Pre-tokenize C4 on the LOGIN node so GPU jobs never need internet.
# Writes <repo>/data/c4_tokens.pt. Idempotent: skips if the cache exists.
#
# Usage:
#   bash prepare_data.sh

set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
CACHE="$REPO_DIR/data/c4_tokens.pt"

if [[ -f "$CACHE" ]]; then
    SIZE="$(du -h "$CACHE" | awk '{print $1}')"
    echo "[skip] $CACHE already exists ($SIZE)"
    exit 0
fi

# Activate env
PY_MODULE="${PY_MODULE_OVERRIDE:-}"
if [[ -z "$PY_MODULE" ]]; then
    for cand in 3.11.6-gcc-13.2.0 3.11.6-gcc-11.3.1 3.10.12-gcc-13.2.0; do
        line="$(module avail "python/$cand" 2>&1 | grep -oE "python/$cand[^ ]*" | head -1 || true)"
        [[ -n "$line" ]] && PY_MODULE="$line" && break
    done
fi
if [[ -n "$PY_MODULE" ]]; then
    module load "$PY_MODULE"
fi
# shellcheck disable=SC1091
source venv/bin/activate
# env_setup.sh handles SSL cert bundle, cache redirection, and C4_CACHE.
source env_setup.sh
echo "[ssl] using CA bundle: $SSL_CERT_FILE"

echo "[step] downloading + tokenizing C4 (this takes ~3 to 10 minutes)"
mkdir -p "$REPO_DIR/data"
python - <<PY
import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE = "$CACHE"
N_DOCS = int(os.environ.get("C4_N_DOCS", "50000"))

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
all_tokens = []
for i, example in enumerate(ds):
    if i >= N_DOCS:
        break
    ids = tokenizer.encode(example["text"], add_special_tokens=False)
    all_tokens.extend(ids)
    if (i + 1) % 5000 == 0:
        print(f"  tokenized {i+1}/{N_DOCS} docs ({len(all_tokens)} tokens)")
tokens = torch.tensor(all_tokens, dtype=torch.int32)
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
torch.save(tokens, CACHE)
print(f"[ok] wrote {CACHE}: {len(all_tokens)} tokens, "
      f"{os.path.getsize(CACHE)/1e6:.1f} MB")
PY

echo "[done] cache at $CACHE"
