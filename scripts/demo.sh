#!/usr/bin/env bash
# A short end-to-end demo: FLM, FMLM, Markov, evaluation, and HTTP service.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 FLM on 8 Gaussians"
python -m flowmap.train --config configs/flm_tiny.yaml --out runs/demo-flm --steps 300

echo "== 2/5 FMLM distillation"
python -m flowmap.train --config configs/fmlm_tiny.yaml --out runs/demo-fmlm \
  --steps 200 --set train.teacher_steps=200

echo "== 3/5 Markov language task"
python -m flowmap.train --config configs/markov_flm.yaml --out runs/demo-markov --steps 200

echo "== 4/5 evaluate the FLM checkpoint"
python -m flowmap.eval --ckpt runs/demo-flm/ckpt.pt --ema

echo "== 5/5 serve and query"
python -m flowmap.serve --ckpt runs/demo-flm/ckpt.pt --port 8140 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 2
curl -s localhost:8140/health
echo
curl -s -X POST localhost:8140/generate \
  -H 'Content-Type: application/json' -d '{"n":4,"steps":8}'
echo
kill $SERVER_PID 2>/dev/null || true
trap - EXIT
echo "demo done"
