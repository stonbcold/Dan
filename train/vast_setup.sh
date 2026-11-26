#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Il manque le token. Fais : export HF_TOKEN=hf_xxx}"
DIR=/workspace/dan
CODE_REPO=jaay211/dan
DATA_REPO=jaay211/dan-pretrain-data

PY=""
for c in python python3 /venv/main/bin/python /opt/conda/bin/python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1; then
        PY=$(command -v "$c")
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Aucun Python avec PyTorch trouve. Essaie : ls /venv /opt/conda ; which -a python3"
    exit 1
fi
if ! command -v python >/dev/null 2>&1 || [ "$(command -v python)" != "$PY" ]; then
    ln -sf "$PY" /usr/local/bin/python
    hash -r
fi
echo "Python utilise : $PY"

python -m pip install -q -U huggingface_hub numpy regex datasets
hf download "$CODE_REPO" --local-dir "$DIR"
hf download "$DATA_REPO" --repo-type dataset --local-dir "$DIR/data/tokenized"
cd "$DIR"

python - <<'EOF'
import json, os, torch
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | capacite {cap} | "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} Go")
for n in ["fineweb", "finemath", "code_translated", "code_textblock"]:
    meta = json.load(open(f"data/tokenized/{n}_meta.json"))
    for split in ["train", "val"]:
        size = os.path.getsize(f"data/tokenized/{n}_{split}.bin") // 2
        assert size == meta["tokens"][split], f"{n}_{split} : {size} != {meta['tokens'][split]}"
print("Donnees completes et coherentes")
EOF

for MB in 8 4 2; do
    echo "===== Test micro_batch=$MB ====="
    if python train/train.py --model dan --out checkpoints/bench --micro_batch "$MB" \
        --total_steps 30 --eval_every 1000000 --eval_batches 2 --log_every 5 \
        --save_every_min 100000 2>&1 | tee bench_$MB.log; then
        if ! grep -q "Error" bench_$MB.log; then
            echo "$MB" > micro_batch.txt
            break
        fi
    fi
done
rm -rf checkpoints/bench

MB=$(cat micro_batch.txt)
echo
echo "micro_batch retenu : $MB"
grep "^etape 30" bench_$MB.log || true
echo
echo "Jour 1 :"
echo "  tmux new -s dan"
echo "  python train/train.py --model dan --micro_batch $MB --max_hours 23.5 2>&1 | tee -a train_dan.log"
echo "Sauvegarde vers Hugging Face (dans un autre terminal) :"
echo "  nohup bash train/vast_sync.sh dan > sync.log 2>&1 &"
