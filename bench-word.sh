#!/bin/bash
# Word model benchmark: Python → Rust, sequential, clean run.
# Both from scratch, no transfer. Same architecture, same data, same GPU.
set -e

echo "=== Word Model Benchmark ==="
echo "Started: $(date)"
echo ""

# --- Python ---
echo "=== Python (120 epochs, batch=32, from scratch) ==="
cd /home/peta/src/fab2s/ai/fbrl/python
rm -rf runs/words/v3-clean
docker compose run --rm -e PYTHONUNBUFFERED=1 --entrypoint python python \
    vision_training.py train_words \
    --config configs/word_v3_clean.yaml \
    --device cuda \
    --checkpoint_interval 50
echo ""
echo "Python finished: $(date)"
echo ""

# --- Rust ---
echo "=== Rust (120 epochs, batch=32, from scratch) ==="
cd /home/peta/src/fab2s/ai/fbrl
rm -rf word/runs/v1
make train-word \
    DATA=../python/data/words \
    WORD_SAVE=runs/v1 \
    BATCH=32 \
    EPOCHS=120 \
    ISOLATION=../python/data/letters \
    MONITOR=0
echo ""
echo "Rust finished: $(date)"
echo ""

echo "=== Benchmark Complete ==="
echo "Python results: python/runs/words/v3-clean/"
echo "Rust results:   word/runs/v1/"
echo "Finished: $(date)"
