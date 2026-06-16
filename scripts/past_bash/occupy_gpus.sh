#!/bin/bash
# Usage:
#   bash scripts/occupy_gpus.sh          # 占所有GPU
#   bash scripts/occupy_gpus.sh 0,1,2,3  # 占指定GPU
#   按 Ctrl+C 释放

GPUS=${1:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')}
GPUS=${GPUS%,}

echo "Occupying GPUs: $GPUS"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
pids=()

for gpu in "${GPU_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python -c "
import torch, signal, sys
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
d = torch.device('cuda:0')
bufs = [torch.empty(256, 1024, 1024, device=d) for _ in range(4)]
a = torch.randn(4096, 4096, device=d)
b = torch.randn(4096, 4096, device=d)
print(f'GPU $gpu: {torch.cuda.memory_allocated(d)/1e9:.1f}GB allocated, computing...')
while True:
    torch.mm(a, b)
" &
    pids+=($!)
done

trap "kill ${pids[*]} 2>/dev/null; echo 'Released all GPUs'; exit 0" INT TERM
echo "PIDs: ${pids[*]}"
echo "Press Ctrl+C to release"
wait
